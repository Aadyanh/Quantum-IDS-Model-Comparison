import math
import torch


def angle_encode(x):
    """Angle encoding RY(pi * x_i)|0> on every qubit: (B, n) values in [0, 1] -> (B, 2^n) product state."""
    a = torch.stack([torch.cos(math.pi * x / 2), torch.sin(math.pi * x / 2)], -1)   # (B, n, 2)
    phi = a[:, 0]
    for i in range(1, x.shape[1]):
        phi = (phi[:, :, None] * a[:, i, None, :]).reshape(len(x), -1)
    return phi
