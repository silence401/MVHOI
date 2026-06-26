import torch 
import torchvision.transforms.functional as F
import random
from einops import rearrange

def random_block_erase(masks, max_ratio=0.25, min_ratio=0.05):
    """
    masks: (N, 1, H, W), values in {0,1}
    在 mask=1 的区域内，随机抹掉一整块矩形区域
    """
    N, _, H, W = masks.shape
    new_masks = masks.clone()

    for i in range(N):
        mask = masks[i, 0]

        if mask.sum() == 0:
            continue

        # 随机 block 尺寸
        erase_area = torch.empty(1).uniform_(min_ratio, max_ratio).item()
        block_h = int(H * erase_area)
        block_w = int(W * erase_area)

        block_h = max(1, block_h)
        block_w = max(1, block_w)

        # 随机中心点（从物体区域采样）
        ys, xs = torch.where(mask)
        idx = torch.randint(len(xs), (1,))
        cy, cx = ys[idx], xs[idx]

        y1 = torch.clamp(cy - block_h // 2, 0, H)
        y2 = torch.clamp(cy + block_h // 2, 0, H)
        x1 = torch.clamp(cx - block_w // 2, 0, W)
        x2 = torch.clamp(cx + block_w // 2, 0, W)

        new_masks[i, 0, y1:y2, x1:x2] = 0

    return new_masks * (masks > 0.5).float()

def blur_and_crop(images, masks, p=0.5):
    b, t, c, h, w = images.shape
    # import pdb; pdb.set_trace()
    images = rearrange(images, 'b t c h w -> (b t) c h w')
    
    masks = rearrange(masks, 'b t h w -> (b t) h w')
    masks = masks.unsqueeze(1)
    # import pdb; pdb.set_trace()
    if torch.rand(1) < p:
        masks = random_block_erase(
            masks,
            min_ratio=0.2,
            max_ratio=0.3
        )
        masks = (masks > 0.5).float()
    # import pdb; pdb.set_trace()
    processed_images = images * masks + (1 - masks) 

    if torch.rand(1) < p:
        k = random.choice([9, 11, 15, 21])
        sigma = random.uniform(3.0, 8.0)
        processed_images = F.gaussian_blur(
            processed_images,
            kernel_size=[k, k],
            sigma=(sigma, sigma)
        )

    processed_images = rearrange(
        processed_images, '(b t) c h w -> b t c h w', t=t
    )
    return processed_images
