import torch


def tv_reg_flow_2d(x: torch.Tensor):
    """
    x: [H, W, 2]
    Returns total variation loss over spatial dimensions
    """
    diff_h = torch.abs(x[:-1, :, :] - x[1:, :, :])  # temporal
    diff_w = torch.abs(x[:, :-1, :] - x[:, 1:, :])  # temporal
    return diff_h.mean() + diff_w.mean()
