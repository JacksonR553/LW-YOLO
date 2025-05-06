import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else tuple(output_size)
        
    def forward(self, x):
        # For common case of global average pooling (output_size=1)
        if self.output_size == (1, 1):
            return F.avg_pool2d(x, kernel_size=x.shape[2:])
        
        # For other cases, use interpolation + pooling
        return F.avg_pool2d(
            F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False),
            kernel_size=1
        )
