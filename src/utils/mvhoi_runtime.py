"""Shared runtime helpers for the MVHOI inference scripts."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
from easydict import EasyDict as edict
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
THIRDPARTY_DIR = REPO_ROOT / "thirdparty"
DISMO_DIR = THIRDPARTY_DIR / "DisMo"

AMP_DTYPE_MAPPING = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}


def ensure_runtime_paths() -> None:
    """Make project-local and third-party imports independent of cwd."""
    for path in reversed((PROJECT_ROOT, THIRDPARTY_DIR, DISMO_DIR)):
        path_str = str(path)
        if not path.exists():
            continue
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)

    torch_hub_dir = REPO_ROOT / ".cache" / "torch"
    torch_hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_hub_dir))


def resolve_project_path(path: Optional[str]) -> Optional[str]:
    """Resolve relative filesystem paths against the project root."""
    if path is None or path == "":
        return path

    expanded_path = os.path.expandvars(os.path.expanduser(str(path)))
    if "://" in expanded_path:
        return expanded_path

    path_obj = Path(expanded_path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((PROJECT_ROOT / path_obj).resolve())


def _get_nested(config: Any, keys: Sequence[str]) -> Any:
    cur = config
    for key in keys:
        if cur is None or key not in cur:
            return None
        cur = cur[key]
    return cur


def _set_nested(config: Any, keys: Sequence[str], value: Any) -> None:
    cur = config
    for key in keys[:-1]:
        cur = cur[key]
    cur[keys[-1]] = value


def _resolve_path_field(config: Any, keys: Sequence[str]) -> None:
    value = _get_nested(config, keys)
    if value in (None, ""):
        return
    _set_nested(config, keys, resolve_project_path(value))


def resolve_config_paths(config: Any) -> Any:
    """Normalize path-like config values used by the MVHOI runtime."""
    for keys in (("model", "object_motion_embedder", "pretrained_path"),):
        _resolve_path_field(config, keys)

    for key in ("checkpoint", "checkpoint_path", "checkpoint_dir", "data_path"):
        _resolve_path_field(config, ("inference", key))

    metadata_path = _get_nested(config, ("inference", "metadata_path"))
    if isinstance(metadata_path, str):
        _set_nested(config, ("inference", "metadata_path"), resolve_project_path(metadata_path))
    elif isinstance(metadata_path, (list, tuple)):
        resolved_paths = [resolve_project_path(path) for path in metadata_path]
        _set_nested(config, ("inference", "metadata_path"), resolved_paths)

    return config


def load_mvhoi_config(config_path: str, overrides: Optional[Iterable[str]] = None) -> edict:
    """Load a YAML config, apply OmegaConf overrides, and normalize runtime paths."""
    config = OmegaConf.load(resolve_project_path(config_path))
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_cli(list(overrides)))
    config_dict = OmegaConf.to_container(config, resolve=True)
    return resolve_config_paths(edict(config_dict))


def import_config_class(dotted_path: str) -> type:
    """Import a class from config, supporting short package-local paths."""
    module_name, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_mvhoi_model(config: Any, device: torch.device, mode: str):
    """Build the MVHOI model."""
    ensure_runtime_paths()
    from model.wrapper import MVHOI3D

    runtime = config.inference
    model_name = config.model.get("model_name", "mvhoi3d-large")
    model = MVHOI3D(model_name=model_name, mode=mode).to(device)
    model.model.modify_head()
    model.model.rgb_head.to(device)
    model.model.add_object_motion_tokenizer(config)
    model.model.object_motion_tokenizer.to(device)

    if runtime.get("use_bf16", False):
        model = model.to(AMP_DTYPE_MAPPING["bf16"])

    return model


def move_batch_to_device(batch: dict, device: torch.device, use_bf16: bool = False) -> dict:
    """Move tensor values to device and optionally cast them to bf16."""
    moved_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    if not use_bf16:
        return moved_batch
    return {
        key: value.to(AMP_DTYPE_MAPPING["bf16"]) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for key, value in moved_batch.items()
    }


def find_latest_checkpoint(load_path: Optional[str]) -> Optional[str]:
    """Return a checkpoint path, or the latest .pt under a checkpoint directory."""
    if not load_path:
        return None
    if os.path.isfile(load_path):
        return load_path
    if not os.path.isdir(load_path):
        return None
    ckpt_files = sorted(file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt"))
    if not ckpt_files:
        return None
    return os.path.join(load_path, ckpt_files[-1])
