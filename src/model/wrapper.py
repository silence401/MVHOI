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
"""High-level MVHOI3D model wrapper used by the inference entrypoints."""

import torch
from easydict import EasyDict as edict
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from typing import Optional

from cfg import create_object, load_config
from registry import MODEL_REGISTRY

torch.backends.cudnn.benchmark = False

SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


class MVHOI3D(nn.Module, PyTorchModelHubMixin):
    """
    MVHOI3D model wrapper class.

    This class wraps the configured MVHOI3D backbone and exposes the batch-level
    `inference_video` path used by this project.
    """

    _commit_hash: Optional[str] = None  # Set by mixin when loading from Hub

    def __init__(self, model_name: str = "mvhoi3d-large", mode: str = "eval", **kwargs):
        """
        Initialize MVHOI3D with the specified preset.

        Args:
            model_name: The name of the model preset to use.
            **kwargs: Additional keyword arguments (currently unused).
        """
        super().__init__()
        self.model_name = model_name
        # Build the underlying network
        self.config = load_config(MODEL_REGISTRY[self.model_name])
        self.model = create_object(self.config)
        self.device = None
        self.model.eval()

    @torch.no_grad()
    def inference_video(self, data):
        """Autoregressive video inference forward pass."""
        output = self.model.inference_video(data)
        result = edict({
            'rgb': output['rgb'],
            'motions': output['motions'],
            'refs': output['refs'],
        })

        return result
