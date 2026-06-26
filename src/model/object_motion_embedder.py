import torch 
from torch import nn
from einops import rearrange, repeat
from functools import reduce
import numpy as np
import cv2
from easydict import EasyDict as edict
from utils.mvhoi_runtime import ensure_runtime_paths
# from RAFT.core.raft import RAFT
# from RAFT.core.utils.utils import InputPadder
# from RAFT.core.utils import flow_viz

ensure_runtime_paths()
from DisMo.dismo.model import MotionExtractor

def viz(img, flo):
    img = img[0].permute(1,2,0).float().cpu().numpy() * 255
    flo = flo[0].permute(1,2,0).float().cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    cv2.imwrite('image.png', img_flo[:, :, [2,1,0]])
    # cv2.waitKey()


class MotionEncoder(nn.Module):
    def __init__(self, flow_channels=2, hidden_dim=128, motion_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(flow_channels, hidden_dim, 7, padding=3),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),  # (B, 128, 1, 1)
        )
        self.fc = nn.Linear(hidden_dim, motion_dim)

    def forward(self, flow):
        feat = self.net(flow)        # (B,128,1,1)
        feat = feat.flatten(1)       # (B,128)
        code = self.fc(feat)         # (B,motion_dim)
        return code


class MotionDecoder(nn.Module):
    def __init__(self, img_channels=3, motion_dim=256, eps=1e-6):
        super().__init__()
        
        # simple UNet encoding
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )

        # 将 motion_code 映射到调制参数
        self.mod_fc = nn.Linear(motion_dim, 128 * 2)

        # decoding
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, img_channels, 3, padding=1),
        )
        self.eps = eps


    def forward(self, img_t, motion_code):   
        x = self.encoder(img_t)   # (B,128,H,W)
        gamma, beta = self.mod_fc(motion_code).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta  = beta[:, :, None, None]

        feat = rms_norm(x, gamma, self.eps) + beta # 调制点
        out = self.decoder(feat)
        return out


class MotionTransfer(nn.Module):
    def __init__(self, config, motion_dim=256):
        super().__init__()
        self.config = config
        # self.raft = torch.hub.load(
        #             'princeton-vl/RAFT',
        #             'raft-large',        # raft-small / raft-large
        #             pretrained=True
        #         )
        args = edict({
            'small': False,
            'mixed_precision': False,
            'alternate_corr': False,
        })

        self.raft = RAFT(args)
        state_dict = torch.load("./RAFT/models/raft-things.pth", map_location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.", "")  # 去掉 prefix
            new_state_dict[new_k] = v
        self.raft.load_state_dict(new_state_dict)
        # model = model.cuda().eval()
        self.freeze_raft()
        self.encoder = MotionEncoder(motion_dim=motion_dim)
        self.decoder = MotionDecoder(motion_dim=motion_dim)
    
    def freeze_raft(self):
        for p in self.raft.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def raft_forward(self, x, y):
        _, flow = self.raft(x, y, iters=20, test_mode=True)
        return flow

    def forward(self, src1, src2, target):
        flow_s = self.raft_forward(src1, src2)
        # 2) 编码成可迁移的 motion_code
        code = self.encoder(flow_s)
        # 3) 用 motion 对 target 做调制
        driven = self.decoder(target, code)
        return code, driven

def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)

class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps

        # 输出 [gamma, beta]
        self.linear = nn.Linear(cond_features, features * 2, bias=False)
        nn.init.zeros_(self.linear.weight)

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        """
        x:    (B, N, C)
        cond: (B, D)
        """
        gamma_beta = self.linear(cond)           # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = gamma[:, None, :] + 1.0           # (B, 1, C)
        beta  = beta[:, None, :]                  # (B, 1, C)

        return rms_norm(x, gamma, self.eps) + beta

class DismoMotion(nn.Module):
    def __init__(self, config):
        super().__init__()
        motion_extractor_params=dict(
            width=1024,
            depth=20,
            d_head=64,
            d_motion=128,
            frame_encoder_params=dict(
                model_version="dinov2_vitl14_reg",
                gradient_last_blocks=2,
            ),
            max_delta_time=4,
            train_resolution=[8, 256, 256],
        )
        self.motion_extractor = MotionExtractor(**motion_extractor_params)
        state_dict = torch.load(config.model.object_motion_embedder.pretrained_path, map_location="cpu")
        self.motion_extractor.load_state_dict(state_dict)
        self.decoder = MotionDecoder(motion_dim=motion_extractor_params['d_motion'])
        self.freeze_motion_extractor()
    
    def freeze_motion_extractor(self):
        for param in self.motion_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, frames, input_frames):
        b, t, c, h, w = frames.shape
        frames = frames * 2 - 1  # [-1, 1]
        with torch.no_grad():
            motion_emb = self.motion_extractor(frames.permute(0, 1, 3, 4, 2))
        motion_emb = rearrange(motion_emb, "b t d -> (b t) d")
        input_frames = rearrange(input_frames, "b t c h w -> (b t) c h w")
        driven = self.decoder(input_frames, motion_emb)

        return motion_emb, driven
        
    def forward_motion(self, frames):
        b, t, c, h, w = frames.shape
        frames = frames * 2 - 1  # [-1, 1]
        with torch.no_grad():
            motion_emb = self.motion_extractor.forward_sliding(frames.permute(0, 1, 3, 4, 2))
        motion_emb = rearrange(motion_emb, "b t d -> (b t) d")
        return motion_emb
    
    def apply_motion(self, input_frame, motion_emb):
        input_frame = rearrange(input_frame, "b t c h w -> (b t) c h w")
        driven = self.decoder(input_frame, motion_emb)
        return driven
