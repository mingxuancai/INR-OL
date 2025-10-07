import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


def mask(encoding, mask_coef):
    mask_coef = 0.4 + 0.6 * mask_coef
    # interpolate to size of encoding
    mask = torch.zeros_like(encoding[0:1])
    mask_ceil = int(np.ceil(mask_coef * encoding.shape[1]))
    mask[:, :mask_ceil] = 1.0

    return encoding * mask


def visualize_flow(flow):
    H, W = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]

    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 0] = ang / 2  # 角度 (0-360) → 色调 (0-180)
    hsv[..., 1] = 255  # 饱和度固定
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 幅度 → 亮度

    plt.figure()
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)
    plt.title("Optical Flow (HSV)")
    plt.axis("off")
    plt.show()


def warp(img, flow, grid):
    # img: [H, W]
    # flow: [H, W, 2]
    # return: [H, W]
    H, W = img.shape
    img = img[None, None, ...]
    grid_warped = grid + flow

    grid_x = 2.0 * grid_warped[..., 0] / (W - 1) - 1
    grid_y = 2.0 * grid_warped[..., 1] / (H - 1) - 1
    grid_norm = torch.stack((grid_x, grid_y), dim=-1)[None]  # [1, H, W, 2]

    warped = F.grid_sample(
        img, grid_norm, mode="bilinear", padding_mode="border", align_corners=True
    )
    return warped[0, 0]
