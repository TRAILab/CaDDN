import torch.nn as nn


class BasicBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Initializes convolutional block for channel reduce
        Args:
            out_channels [int]: Number of output channels of convolutional block
            **kwargs [Dict]: Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """
        Applies convolutional block
        Args:
            features [torch.Tensor(B, C_in, H, W)]: Input features
        Returns:
            x [torch.Tensor(B, C_out, H, W)]: Output features
        """
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x
