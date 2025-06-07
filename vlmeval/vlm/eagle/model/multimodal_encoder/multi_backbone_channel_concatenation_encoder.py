# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .convnext_encoder import ConvNextVisionTower
from .hr_clip_encoder import HRCLIPVisionTower
from .vision_models.eva_vit import EVAVITVisionTower
from .sam_encoder import SAMVisionTower
from .pix2struct_encoder import Pix2StructLargeVisionTower
from transformers import CLIPImageProcessor
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from copy import deepcopy
import random
import math
from loguru import logger


class MultiBackboneChannelConcatenationVisionTower(nn.Module):
    def __init__(self, vision_tower, args, grid_size=32):
        super().__init__()

        self.is_loaded = False
        self.grid_size = grid_size
        self.num_tokens = self.grid_size**2

        vision_tower_name_list = vision_tower.split(";")
        self.input_image_size = 1024  # hardcode

        self.mask_vision_tower_index = [0, 4]
        logger.info(
            f"Masking features from vision tower {self.mask_vision_tower_index}"
        )

        self.load_vision_towers(vision_tower_name_list, args)

    def load_vision_towers(self, vision_tower_name_list, args):
        self.vision_towers = nn.ModuleList()
        for index, name in enumerate(vision_tower_name_list):
            logger.info(f"index: {index}, Loading vision tower: {name}")
            # mask vision tower
            if index in self.mask_vision_tower_index:
                logger.info(f"Skipping vision tower: {name} with index: {index}")
                self.vision_towers.append(None)
                continue

            if name == "det-1024":
                det_args = deepcopy(args)
                det_args.input_image_size = 1024
                det_args.freeze_vision = False
                det_args.vision_tower_pretrained_from = (
                    "checkpoints/pretrained_models/eva02_L_coco_det_sys_o365.pth"
                )
                det_vision_tower = EVAVITVisionTower("eva02-l-16", det_args)
                det_vision_tower.load_model()
                self.vision_towers.append(det_vision_tower)

            elif name == "convnext-1024":
                ## ConvNeXt
                convnext_args = deepcopy(args)
                convnext_args.freeze_vision = False
                convnext_args.input_image_size = 1024
                convnext_vision_tower = "convnext_xxlarge.clip_laion2b_soup"  # hardcode
                convnext_vision_tower = ConvNextVisionTower(
                    convnext_vision_tower, convnext_args
                )
                convnext_vision_tower.load_model()
                self.vision_towers.append(convnext_vision_tower)

            elif name == "sam-1024":
                sam_args = deepcopy(args)
                sam_args.freeze_vision = False
                sam_args.input_image_size = 1024
                sam_args.add_pixel_shuffle = True
                sam_vision_tower = SAMVisionTower("SAM-L", sam_args)
                sam_vision_tower.load_model()
                self.vision_towers.append(sam_vision_tower)

            elif name == "pix2struct-1024":
                pix_args = deepcopy(args)
                # pix_args.freeze_vision = True
                pix_args.input_image_size = 1024
                pix_args.freeze_vision = False
                pix_args.do_resize = True
                pix_args.de_normalize = True
                pix_vision_tower = Pix2StructLargeVisionTower(
                    "pix2struct-large", pix_args
                )
                pix_vision_tower.load_model()
                self.vision_towers.append(pix_vision_tower)

            elif name == "clip-448":
                clip_args = deepcopy(args)
                clip_args.input_image_size = 336  # actually 448, will have no effect
                clip_args.freeze_vision = False
                clip_vision_tower = HRCLIPVisionTower(
                    "openai/clip-vit-large-patch14-336", clip_args
                )
                clip_vision_tower.load_model()
                self.vision_towers.append(clip_vision_tower)

        # a hardcode here, so we always use convnext in the vision encoder mixture
        self.image_processor = CLIPImageProcessor()
        self.image_processor.size = 1024
        self.image_processor.crop_size = {"height": 1024, "width": 1024}
        self.is_loaded = True

        self.index2shape = {
            0: [1, 1024, 1024],
            1: [1, 1024, 3072],
            2: [1, 1024, 1024],
            3: [1, 1024, 1024],
            4: [1, 1024, 1536],
        }

    def load_model(self):
        assert self.is_loaded, (
            "All the vision encoders should be loaded during initialization!"
        )

    def forward(self, x):
        features = []
        b, c, h, w = x.shape
        for index, vision_tower in enumerate(self.vision_towers):
            # use dummy feature for masked vision tower
            if index in self.mask_vision_tower_index:
                tensor_shape = self.index2shape[index]
                tensor_shape[0] = b
                dummy_feature = torch.zeros(tensor_shape, dtype=x.dtype).to(x.device)
                features.append(dummy_feature)
                continue

            if vision_tower.input_image_size != self.input_image_size:
                resized_x = F.interpolate(
                    x.float(),
                    size=(vision_tower.input_image_size, vision_tower.input_image_size),
                    mode="bilinear",
                    align_corners=True,
                ).to(dtype=x.dtype)
            else:
                resized_x = x
            feature = vision_tower(resized_x)
            if len(feature.shape) == 3:  # b, n, c
                b, n, c = feature.shape
                if n == self.num_tokens:
                    features.append(feature)
                    continue

                w = h = int(n**0.5)
                feature = feature.transpose(1, 2).reshape(b, c, h, w)
            else:
                b, c, h, w = feature.shape

            if w != self.grid_size:
                feature = F.interpolate(
                    feature.float(),
                    size=(self.grid_size, self.grid_size),
                    mode="bilinear",
                    align_corners=True,
                ).to(dtype=x.dtype)

            feature = feature.flatten(2, 3).transpose(1, 2)
            features.append(feature)

        features = torch.cat(features, dim=-1)

        return features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.clip_vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.clip_vision_tower.parameters()).device

    @property
    def config(self):
        assert NotImplementedError
        pass

    @property
    def hidden_size(self):
        return sum([_.hidden_size for _ in self.vision_towers])

    @property
    def num_patches(self):
        return self.num_tokens
