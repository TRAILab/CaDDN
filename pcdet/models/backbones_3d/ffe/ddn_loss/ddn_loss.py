import torch
import torch.nn as nn

from .loss_functions import loss_functions
from .balancer import Balancer
from pcdet.utils import depth_utils


class DDNLoss(nn.Module):

    def __init__(self,
                 weight,
                 alpha,
                 gamma,
                 disc_cfg,
                 fg_bg_balancer_cfg,
                 downsample_factor):
        """
        Initializes DCNLoss module
        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            gamma [float]: Gamma value for Focal Loss
            num_depths [int]: Number of depth bins D
            depth_min [float]: Minimum depth value for classification
            depth_max [float]: Maximum depth value for classification
            fg_bg_balancer_cfg [EasyDict]: Foreground/Background balancer config
            downsample_factor [int]: Depth map downsample factor. Only needed for foreground/background balancing
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.disc_cfg = disc_cfg
        self.balancer = Balancer(downsample_factor=downsample_factor,
                                 fg_weight=fg_weight,
                                 bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = loss_functions[func](alpha=self.alpha, gamma=self.gamma, reduction="none")
        self.weight = weight

    def forward(self, depth_logits, depth_map, gt_boxes_2d=None):
        """
        Gets DCN loss
        Args:
            depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
            depth_map: torch.Tensor(B, H, W)]: Depth map [m]
            gt_boxes_2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
        Returns:
            loss [torch.Tensor(1)]: Depth classification network Loss
            tb_dict [dict[float]]: All losses to log in tensorboard
        """
        tb_dict = {}

        # Bin depth map to create target
        depth_target = depth_utils.bin_depths(depth_map, **self.disc_cfg, target=True)

        # Compute loss
        loss = self.loss_func(depth_logits, depth_target)

        # Compute foreground/background balancing
        loss, tb_dict = self.fg_bg_balancer(loss=loss, gt_boxes_2d=gt_boxes_2d)

        # Final loss
        loss *= self.weight
        tb_dict.update({"dcn_loss": loss.item()})

        return loss, tb_dict
