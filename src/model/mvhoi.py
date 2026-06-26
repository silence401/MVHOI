# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
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
# limitations under the License.
# Modifications Copyright (c) 2026 Baidu

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from copy import deepcopy
from addict import Dict
from einops import rearrange, repeat
from omegaconf import OmegaConf

from cfg import create_object
from model.utils.transform import pose_encoding_to_extri_intri
from utils.geometry import affine_inverse
from utils.augmentation import blur_and_crop
from model.object_motion_embedder import DismoMotion

def init_weights(module, std=0.02):
    """Initialize weights for linear and embedding layers.
    
    Args:
        module: Module to initialize
        std: Standard deviation for normal initialization
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)

class MVHOI3DNet(nn.Module):
    """
    MVHOI3D Net
    """

    NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Patch size for feature extraction
    PATCH_SIZE = 14

    def __init__(self, net, head, cam_dec=None, cam_enc=None):
        """
        Initialize MVHOI3DNet with given yaml-initialized configuration.
        """
        super().__init__()
        self.backbone = net if isinstance(net, nn.Module) else create_object(_wrap_cfg(net))
        self.head = head if isinstance(head, nn.Module) else create_object(_wrap_cfg(head))
        self.cam_dec, self.cam_enc = None, None
        if cam_dec is not None:
            self.cam_dec = (
                cam_dec if isinstance(cam_dec, nn.Module) else create_object(_wrap_cfg(cam_dec))
            )
            self.cam_enc = (
                cam_enc if isinstance(cam_enc, nn.Module) else create_object(_wrap_cfg(cam_enc))
            )
        # self.modify_head()
        # self.add_pose_tokenizer()

    def modify_head(self,):
        self.rgb_head = deepcopy(self.head)
        self.rgb_head.activation = "sigmoid"
        self.rgb_head.scratch.output_conv1.apply(init_weights)
        head_features_2 = self.rgb_head.scratch.output_conv2[0].out_channels
        self.rgb_head.scratch.output_conv2[-1] = nn.Conv2d(head_features_2, 4, kernel_size=1, stride=1, padding=0)
        self.rgb_head.scratch.output_conv2.apply(init_weights)
        del self.head

    def modify_cam_token(self):
        with torch.no_grad():
            B, S, C = self.backbone.pretrained.camera_token.shape
            device = self.backbone.pretrained.camera_token.device
            dtype = self.backbone.pretrained.camera_token.dtype

            # 新的 toke[n
            cam_token_new = torch.zeros(B, 6, C, device=device, dtype=dtype)

            # 复制内容
            cam_token_new[:, :4] = self.backbone.pretrained.camera_token[:, 0:1]  # ref view × 4
            cam_token_new[:, 4:] = self.backbone.pretrained.camera_token[:, 1:2]  # input + target view

        # 替换为 nn.Parameter
        self.backbone.pretrained.camera_token = nn.Parameter(cam_token_new)
        self.backbone.pretrained.camera_token.requires_grad_(True)

    def add_object_motion_tokenizer(self, config):
        self.object_motion_tokenizer = DismoMotion(config)
        # self.object_tokenizer.apply(init_weights)
    
    @torch.no_grad()
    def inference_video(self,
                        input_data,
                        extrinsics: torch.Tensor | None = None,
                        intrinsics: torch.Tensor | None = None,
                        export_feat_layers: list[int] | None = []):
        
        motion_frames = input_data['obj_frames_motion'] #[B, num_frames, C, H, W]
        ref_images = input_data['mv_images'] #[B, 4, C, H, W]
        raw_frames = input_data.get('obj_frames', None)
        if raw_frames is not None:
            _, _, _, h, w = raw_frames.shape
            raw_initial_frame = raw_frames[:, 0, ...]
            frames = self.NORMALIZE(raw_frames)
        elif input_data.get('first_frame', None) is not None:
            raw_initial_frame = input_data['first_frame']
            _, _, h, w = raw_initial_frame.shape
            frames = None
        else:
            raw_initial_frame = ref_images[:, 0, ...]
            _, _, h, w = raw_initial_frame.shape
            frames = None

        motion_num = int(motion_frames.shape[1] - input_data['delta_time'][0].item())
        ref_images = repeat(ref_images, "b t c h w -> (b r) t c h w", r=motion_num)

        ref_images = self.NORMALIZE(ref_images)
        ref_images = ref_images[0].unsqueeze(0)

        motion_emb = self.object_motion_tokenizer.forward_motion(motion_frames)
        output_frames = []
        output_motions = []
        output_refs = []
        for i in range((motion_frames.shape[1] - 1) // 4):
            if i == 0:
                if frames is not None:
                    input_frame = frames[:, i*4, ...]
                    output_frame = input_data['obj_frames'][:, i*4, ...]
                else:
                    input_frame = self.NORMALIZE(raw_initial_frame)
                    output_frame = raw_initial_frame
                input_frame = input_frame.unsqueeze(0)
                output_frames.append(output_frame.unsqueeze(0))

            output_motions.append(motion_frames[:, i*4, ...])
            if raw_frames is not None:
                output_refs.append(input_data['obj_frames'][:, i*4, ...].unsqueeze(0))
            else:
                output_refs.append(raw_initial_frame.unsqueeze(0))

            motion_frame = self.object_motion_tokenizer.apply_motion(input_frame, motion_emb[i*4].unsqueeze(0))
            feats, aux_feats = self.backbone(
                    torch.cat([input_frame, ref_images], dim=1), cam_token=None, export_feat_layers=export_feat_layers, obj_pose=motion_frame)
            
            #To float32 for rgb_head
            target_view_feats = tuple(
                (feat[0][:, -1:].to(torch.float32), feat[1][:, -1:].to(torch.float32)) for feat in feats if len(feat) >= 2
            )
            with torch.autocast(device_type=input_frame.device.type, enabled=False):
                tmp = self._process_rgb_head(target_view_feats, h, w)
                result_frame = tmp['rgb'].permute(0, 1, 4, 2, 3)
                output_frames.append(result_frame)
                input_frame = self.NORMALIZE(result_frame)

        output_refs.append(torch.zeros_like(output_refs[0]))
        output_motions.append(motion_frames[:, (motion_frames.shape[1] - 1) // 4 * 4, ...])
        output_frames = torch.cat(output_frames, dim=0)
        output_motions = torch.stack(output_motions, dim=0)
        output_refs = torch.cat(output_refs, dim=0)

        output = {
            'rgb': output_frames,
            'motions': output_motions,
            'refs': output_refs,
        }
        return output

        # input_frames = rearrange(input_frames, "b (t r) c h w -> (b t) r c h w", r=1)

    def step(self,
            input_data,
            extrinsics: torch.Tensor | None = None,
            intrinsics: torch.Tensor | None = None,
            export_feat_layers: list[int] | None = [],
            ):

        frames = input_data['obj_frames']  # [B, num_frames, C, H, W]
        masks = input_data['obj_frames_masks']
        b, num_frames, c, h, w = frames.shape
        motion_num = int(num_frames - input_data['delta_time'][0].item())

        motion_frames = input_data['obj_frames_motion']  # [B, num_frames, C, H, W]
        ref_images = input_data['mv_images']  # [B, 4, C, H, W]
        ref_images = repeat(ref_images, "b t c h w -> (b r) t c h w", r=motion_num)

        ref_images = self.NORMALIZE(ref_images)
        # blur input images for more focus on ref images
        input_frames = frames[:, :motion_num, ...]
        input_masks = masks[:, :motion_num, ...]
        input_frames = blur_and_crop(input_frames, input_masks, p=0.5)
        aug_input_frames = input_frames.clone()
        input_frames = self.NORMALIZE(input_frames)
        target_frames = frames[:, (num_frames - motion_num):, ...]
        target_frames = self.NORMALIZE(target_frames)

        #with torch.autocast(device_type=input_image.device.type, enabled=False):
        motion_emb, motion_frames = self.object_motion_tokenizer(motion_frames, input_frames)
        input_frames = rearrange(input_frames, "b (t r) c h w -> (b t) r c h w", r=1)
        target_frames = rearrange(target_frames, "b (t r) c h w -> (b t) r c h w", r=1)

        feats, aux_feats = self.backbone(
            torch.cat([input_frames, ref_images], dim=1), cam_token=None, export_feat_layers=export_feat_layers, obj_pose=motion_frames)
        target_view_feats = tuple(
            (feat[0][:, -1:], feat[1][:, -1:]) for feat in feats if len(feat) >= 2
        )
        with torch.autocast(device_type=input_frames.device.type, enabled=False):
            output = self._process_rgb_head(target_view_feats, h, w)
        output['aug_input_frames'] = aug_input_frames
        return output

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input images (B, N, 3, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4) - unused
            intrinsics: Camera intrinsics (B, N, 3, 3) - unused
            feat_layers: List of layer indices to extract features from

        Returns:
            Dictionary containing predictions and auxiliary features
        """
        # Extract features using backbone
        if extrinsics is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None

        feats, aux_feats = self.backbone(
            x, cam_token=cam_token, export_feat_layers=export_feat_layers
        )
        H, W = x.shape[-2], x.shape[-1]

        # Process features through the configured prediction head.
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self._process_model_head(feats, H, W)

        # Extract auxiliary features if requested
        output.aux = self._extract_auxiliary_features(aux_feats, export_feat_layers, H, W)

        return output

    def _process_model_head(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Process features through the configured prediction head."""
        return self.head(feats, H, W, patch_start_idx=0)

    def _process_rgb_head(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Process features through the RGB prediction head."""
        output = self.rgb_head(feats, H, W, patch_start_idx=0)
        if "depth" in output:
            output["rgb"] = output.pop("depth")
        if "depth_conf" in output:
            output["rgb_conf"] = output.pop("depth_conf")
        return output

    def _process_camera_estimation(
        self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
        if self.cam_dec is not None:
            pose_enc = self.cam_dec(feats[-1][1])
            # Remove ray information as it's not needed for pose estimation
            if "ray" in output:
                del output.ray
            if "ray_conf" in output:
                del output.ray_conf

            # Convert pose encoding to extrinsics and intrinsics
            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            output.extrinsics = affine_inverse(c2w)
            output.intrinsics = ixt

        return output

    def _extract_auxiliary_features(
        self, feats: list[torch.Tensor], feat_layers: list[int], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Extract auxiliary features from specified layers."""
        aux_features = Dict()
        assert len(feats) == len(feat_layers)
        for feat, feat_layer in zip(feats, feat_layers):
            # Reshape features to spatial dimensions
            feat_reshaped = feat.reshape(
                [
                    feat.shape[0],
                    feat.shape[1],
                    H // self.PATCH_SIZE,
                    W // self.PATCH_SIZE,
                    feat.shape[-1],
                ]
            )
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped

        return aux_features
