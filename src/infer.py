# Copyright (c) 2025 Baidu Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import argparse
import csv
import json
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from einops import rearrange
from PIL import Image
from rich import print
from tqdm import tqdm

from utils.mvhoi_runtime import (
    AMP_DTYPE_MAPPING,
    build_mvhoi_model,
    ensure_runtime_paths,
    find_latest_checkpoint,
    load_mvhoi_config,
    move_batch_to_device,
    resolve_project_path,
)

ensure_runtime_paths()


def runtime_config(config: Any) -> Any:
    """Return the inference runtime config."""
    return config.inference


def load_inference_config(config_path: str) -> Any:
    """Load config for inference."""
    return load_mvhoi_config(config_path)


def select_checkpoint_path(config: Any, checkpoint: Optional[str] = None) -> Optional[str]:
    """Select an explicit checkpoint, then config checkpoint, then latest in directory."""
    if checkpoint:
        return resolve_project_path(checkpoint)

    runtime = runtime_config(config)
    for key in ("checkpoint", "checkpoint_path"):
        value = runtime.get(key, "")
        if value:
            return resolve_project_path(value)

    checkpoint_dir = runtime.get("checkpoint_dir", "")
    if checkpoint_dir:
        return find_latest_checkpoint(resolve_project_path(checkpoint_dir))
    return None


def load_model_weights(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Load model weights from a checkpoint file."""
    print(f"Loading model weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    status = model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded with status: {status}")


def load_metadata(metadata_path: Union[str, Sequence[str]]) -> List[Dict[str, Any]]:
    """Load inference records from CSV, JSON, or JSONL metadata."""
    if isinstance(metadata_path, (list, tuple)):
        records = []
        for path in metadata_path:
            records.extend(load_metadata(path))
        return records

    if metadata_path.endswith(".json"):
        with open(metadata_path, "r") as fp:
            return json.load(fp)
    if metadata_path.endswith(".jsonl"):
        with open(metadata_path, "r") as fp:
            return [json.loads(line) for line in fp if line.strip()]

    with open(metadata_path, "r") as fp:
        return list(csv.DictReader(fp))


def resolve_data_path(path: str, data_path: str) -> str:
    """Resolve a metadata path against the configured data root."""
    if path is None or path == "":
        return path
    path = os.path.expanduser(os.path.expandvars(str(path)))
    if "://" in path or osp.isabs(path):
        return path
    return osp.join(data_path, path)


def get_record_string(record: Dict[str, Any], key: str) -> str:
    """Return a stripped metadata field value, or an empty string."""
    value = record.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def choose_frame_ids(video_len: int, num_frames: int, skip_frames: int, frame_step: int) -> List[int]:
    """Choose deterministic inference frame ids and clamp short videos to the last frame."""
    if video_len <= 0:
        raise ValueError("Cannot read frames from an empty video")
    frame_ids = [skip_frames + idx * frame_step for idx in range(num_frames)]
    return [min(max(frame_id, 0), video_len - 1) for frame_id in frame_ids]


def load_video_pair(video_path: str, mask_path: str, num_frames: int, skip_frames: int, frame_step: int):
    """Load synchronized RGB video and mask frames as TCHW tensors."""
    from torchvision.io import read_video

    video_thwc, _, _ = read_video(video_path, pts_unit="sec", output_format="THWC")
    mask_thwc, _, _ = read_video(mask_path, pts_unit="sec", output_format="THWC")
    max_len = min(video_thwc.shape[0], mask_thwc.shape[0])
    frame_ids = choose_frame_ids(max_len, num_frames, skip_frames, frame_step)
    idx = torch.as_tensor(frame_ids, dtype=torch.long)
    video = video_thwc.index_select(0, idx).permute(0, 3, 1, 2).contiguous()
    mask = mask_thwc.index_select(0, idx).permute(0, 3, 1, 2).contiguous()
    return video, mask


def resize_and_pad_object(frame: torch.Tensor, mask: torch.Tensor, target_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resize one cropped object frame/mask to a square canvas with white background."""
    _, height, width = frame.shape
    scale = target_size / max(height, width)
    new_height = max(int(height * scale), 1)
    new_width = max(int(width * scale), 1)

    frame = F.interpolate(frame.unsqueeze(0), size=(new_height, new_width), mode="bilinear", align_corners=False)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(new_height, new_width), mode="bilinear", align_corners=False)

    pad_height = target_size - new_height
    pad_width = target_size - new_width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    frame = F.pad(frame, (pad_left, pad_right, pad_top, pad_bottom), value=1.0).squeeze(0)
    mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), value=0.0).squeeze(0).squeeze(0)
    mask = (mask > 0.5).float()
    frame = frame * mask.unsqueeze(0) + torch.ones_like(frame) * (1.0 - mask.unsqueeze(0))
    return frame, mask


def crop_object_by_mask(frames: torch.Tensor, masks: torch.Tensor, target_size: int, margin: float = 0.1):
    """Crop each frame around the object mask and resize to target size."""
    if masks.ndim == 4:
        masks = masks[:, 0]

    cropped_frames = []
    cropped_masks = []
    for frame, mask in zip(frames, masks):
        ys, xs = torch.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Empty object mask in input video")

        x1, x2 = xs.min().item(), xs.max().item()
        y1, y2 = ys.min().item(), ys.max().item()
        width = int((x2 - x1 + 1) * (1.0 + margin))
        height = int((y2 - y1 + 1) * (1.0 + margin))
        cx = int(round((x1 + x2) / 2))
        cy = int(round((y1 + y2) / 2))

        crop_x1 = max(cx - width // 2, 0)
        crop_y1 = max(cy - height // 2, 0)
        crop_x2 = min(crop_x1 + width, frame.shape[-1])
        crop_y2 = min(crop_y1 + height, frame.shape[-2])

        cropped_frame = frame[:, crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        out_frame, out_mask = resize_and_pad_object(cropped_frame, cropped_mask, target_size)
        cropped_frames.append(out_frame)
        cropped_masks.append(out_mask)

    return torch.stack(cropped_frames, dim=0), torch.stack(cropped_masks, dim=0)


def load_rgb_image(path: str, image_size: int) -> torch.Tensor:
    """Load one RGB image as CHW tensor in [0, 1]."""
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = min(image_size / width, image_size / height)
    resize_width = max(round(width * scale), 1)
    resize_height = max(round(height * scale), 1)
    image = TF.resize(image, (resize_height, resize_width))

    pad_left = (image_size - resize_width) // 2
    pad_top = (image_size - resize_height) // 2
    pad_right = image_size - resize_width - pad_left
    pad_bottom = image_size - resize_height - pad_top
    image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=255)
    return TF.to_tensor(image)


def build_inference_batch(record: Dict[str, Any], config: Any) -> Dict[str, Any]:
    """Build the model input dict directly from one metadata record."""
    runtime = runtime_config(config)
    data_path = runtime.data_path
    image_size = int(runtime.image_size)
    motion_size = int(config.model.object_motion_embedder.image_size)
    frame_step = int(runtime.frame_step)

    object_name = get_record_string(record, "obj_name")
    if not object_name:
        raise ValueError("Metadata record is missing obj_name")
    object_image_names = [name.strip() for name in runtime.object_image_names.split(",") if name.strip()]
    object_images = [
        load_rgb_image(osp.join(data_path, object_name, f"{name}.jpg"), image_size)
        for name in object_image_names
    ]
    object_images = torch.stack(object_images, dim=0)

    video_value = get_record_string(record, "video_path")
    mask_value = get_record_string(record, "obj_mask_path")
    if bool(video_value) != bool(mask_value):
        raise ValueError("video_path and obj_mask_path must be provided together")

    object_frames = None
    if video_value and mask_value:
        video_path = resolve_data_path(video_value, data_path).replace("clips_crf1", "clips")
        mask_path = resolve_data_path(mask_value, data_path)
        video_frames, object_masks = load_video_pair(
            video_path,
            mask_path,
            int(runtime.num_frames),
            int(runtime.skip_frames),
            frame_step,
        )
        video_frames = video_frames.float() / 255.0
        object_masks = (object_masks.float() / 255.0 > 0.5).float()
        object_frames, _ = crop_object_by_mask(video_frames, object_masks, image_size)

    motion_path = resolve_data_path(record["motion_path"], data_path).replace("clips_crf1", "clips")
    motion_mask_path = resolve_data_path(record["motion_mask_path"], data_path)

    motion_frames, motion_masks = load_video_pair(
        motion_path,
        motion_mask_path,
        int(runtime.num_frames),
        int(runtime.skip_frames),
        frame_step,
    )
    motion_frames = motion_frames.float() / 255.0
    motion_masks = (motion_masks.float() / 255.0 > 0.5).float()
    object_motion_frames, _ = crop_object_by_mask(motion_frames, motion_masks, image_size)
    object_motion_frames = F.interpolate(
        object_motion_frames,
        size=(motion_size, motion_size),
        mode="bilinear",
        align_corners=False,
    )

    first_frame = None
    first_frame_path = get_record_string(record, "first_frame_path")
    if first_frame_path:
        first_frame_path = resolve_data_path(first_frame_path, data_path)
        first_frame = load_rgb_image(first_frame_path, image_size)
        if object_frames is not None:
            object_frames[0] = first_frame

    batch = {
        "obj_frames_motion": object_motion_frames.unsqueeze(0),
        "mv_images": object_images.unsqueeze(0),
        "delta_time": torch.tensor([4], dtype=torch.long),
        "video_id": [record.get("video_id", "sample")],
    }
    if object_frames is not None:
        batch["obj_frames"] = object_frames.unsqueeze(0)
    if first_frame is not None:
        batch["first_frame"] = first_frame.unsqueeze(0)
    return batch


def to_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    """Convert BCHW tensor to a displayable RGB tensor."""
    channels = tensor.size(1)
    if channels == 1:
        return tensor.repeat(1, 3, 1, 1)
    if channels == 2:
        return torch.cat([tensor, tensor.new_zeros(tensor.size(0), 1, tensor.size(2), tensor.size(3))], dim=1)
    if channels > 3:
        return tensor[:, :3]
    return tensor


def save_video_preview(output_dir: str, result: Any, step: int) -> None:
    """Save a compact refs/prediction/motion preview for one inference batch."""
    images = result.rgb
    motions = result.motions
    refs = result.get("refs", None)
    if images is None:
        return

    batch_size, view_count, _, height, width = images.shape
    images = images.reshape(batch_size * view_count, -1, height, width)
    images = to_three_channels(images)

    panels = []
    if refs is not None:
        refs = refs.reshape(batch_size * view_count, -1, height, width)
        panels.append(to_three_channels(refs))

    panels.append(images)

    if motions is not None:
        motions = motions.reshape(batch_size * view_count, -1, motions.size(-2), motions.size(-1))
        motions = F.interpolate(motions, size=(height, width), mode="bilinear", align_corners=False)
        panels.append(to_three_channels(motions))

    preview = torch.cat(panels, dim=3).detach().float().cpu()
    preview = rearrange(preview, "(b v) c h w -> (b h) (v w) c", b=batch_size, v=view_count)
    preview = (preview.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(preview).save(os.path.join(output_dir, f"preview_{step:06d}.jpg"))


class Inferencer:
    """Minimal DisMo video inferencer."""

    def __init__(self, config: Any, checkpoint: Optional[str], device: str = "cuda") -> None:
        self.config = config
        self.runtime = runtime_config(config)
        self.device = torch.device(device)
        self.checkpoint_path = select_checkpoint_path(config, checkpoint)
        self.records = load_metadata(self.runtime.metadata_path)
        self._init_model()

    def _init_model(self) -> None:
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        model = build_mvhoi_model(self.config, self.device, mode="eval")
        load_model_weights(model, self.checkpoint_path)
        model.eval()
        self.model = model

    def _forward(self, batch: dict) -> Any:
        amp_dtype = AMP_DTYPE_MAPPING[self.runtime.get("amp_dtype", "bf16")]
        use_amp = self.runtime.get("use_amp", True) and self.device.type == "cuda"
        with torch.autocast(enabled=use_amp, device_type=self.device.type, dtype=amp_dtype):
            return self.model.inference_video(batch)

    @torch.no_grad()
    def run(self, output_dir: Optional[str] = None) -> None:
        output_dir = resolve_project_path(output_dir or self.runtime.get("output_dir", "inference_outputs"))
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        for step, record in enumerate(tqdm(self.records, desc="Inference")):
            data = build_inference_batch(record, self.config)
            batch = move_batch_to_device(
                data,
                self.device,
                use_bf16=self.runtime.get("use_bf16", False),
            )
            result = self._forward(batch)

            video_id = data["video_id"][0]
            sample_dir = os.path.join(output_dir, str(video_id))
            os.makedirs(sample_dir, exist_ok=True)
            save_video_preview(sample_dir, result, step)

        print(f"Inference completed. Results saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DisMo video inference")
    parser.add_argument("--config", type=str, default="configs/infer_config_dismo.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for inference outputs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use, e.g. cuda or cpu")
    return parser.parse_args()


def default_video_output_dir(config: Any) -> str:
    runtime = runtime_config(config)
    return resolve_project_path(runtime.get("output_dir", "inference_outputs"))


def validate_device(device: str) -> None:
    """Fail early with a clear error when CUDA is requested but unavailable."""
    if not device.startswith("cuda"):
        return
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        "CUDA device was requested, but torch.cuda.is_available() is False. "
        f"torch={torch.__version__}, torch_cuda={torch.version.cuda}. "
        "Check that the PyTorch CUDA wheel matches the NVIDIA driver, or run with --device cpu."
    )


def run_inference(
    config_path: str,
    checkpoint: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda",
) -> None:
    config = load_inference_config(config_path)
    runtime = runtime_config(config)
    validate_device(device)
    num_threads = runtime.get("num_threads", 1)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

    use_tf32 = runtime.get("use_tf32", True)
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    inferencer = Inferencer(config, checkpoint=checkpoint, device=device)
    inferencer.run(output_dir=output_dir)


def main() -> None:
    args = parse_args()
    run_inference(
        config_path=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
