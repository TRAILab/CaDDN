import torch
import torch.nn as nn
import torch.nn.functional as F


class Sampler(nn.Module):

    def __init__(self, mode="bilinear", padding_mode="zeros"):
        """
        Initializes module
        Args:
            mode [string]: Sampling mode [bilinear/nearest]
            padding_mode [string]: Padding mode for outside grid values [zeros/border/reflection]
        """
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, input_features, grid):
        """
        Samples input using sampling grid
        Args:
            input_features [torch.Tensor(N, C, H_in, W_in)]: Input feature maps
            grid [torch.Tensor(N, H_out, W,_out 2)]: Sampling grids for image features
        Returns
            output_features [torch.Tensor(N, C, H_out, W_out)]: Output feature maps
        """
        # Sample from grid
        output = F.grid_sample(input=input_features, grid=grid, mode=self.mode, padding_mode=self.padding_mode)
        return output
