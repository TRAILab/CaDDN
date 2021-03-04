import torch.nn as nn
import torch.nn.functional as F

from pcdet.models.backbones_3d.ffe import ddn
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D


class DepthFFE(nn.Module):

    def __init__(self, model_cfg):
        """
        Initialize depth classification network
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        self.disc_cfg = model_cfg.DISCRETIZE

        # Create classification network
        ddn_cfg = model_cfg.DDN
        self.ddn = ddn.__all__[ddn_cfg.NAME](
            num_classes=model_cfg.DISCRETIZE["num_bins"] + 1,
            backbone_name=ddn_cfg.BACKBONE_NAME,
            **ddn_cfg.ARGS
        )

        # Create channel reduce module
        self.channel_reduce = BasicBlock2D(**model_cfg.CHANNEL_REDUCE)

        self.loss_module = ""
        self.forward_ret_dict = {}

    def get_output_feature_dim(self):
        return self.channel_reduce.out_channels

    def forward(self, batch_dict, **kwargs):
        """
        Predicts depths and creates image depth feature volume using depth classification scores
        Args:
            images [torch.Tensor(N, 3, H_in, W_in)]: Input images
        Returns:
            frustum_features [torch.Tensor(N, C, D, H_out, W_out)]: Image depth features
            depth_logits [torch.Tensor(N, D, H_, W_out)]:
        """
        # Pixel-wise depth classification
        ddn_result = self.ddn(images)
        image_features = ddn_result["features"]
        depth_logits = ddn_result["logits"]

        # Channel reduce
        if self.channel_reduce is not None:
            image_features = self.channel_reduce(image_features)

        # Create image feature plane-sweep volume
        frustum_features = self.create_frustum_features(image_features=image_features,
                                                        depth_logits=depth_logits)

        if self.training:
            self.forward_ret_dict["depth_map"] = batch_dict["depth_map"]
            self.forward_ret_dict["depth_logits"] = batch_dict["depth_logits"]

        return frustum_features

    def create_frustum_features(self, image_features, depth_logits):
        """
        Create image depth feature volume by multiplying image features with depth classification scores
        Args:
            image_features [torch.Tensor(N, C, H, W)]: Image features
            depth_logits [torch.Tensor(N, D, H, W)]: Depth classification logits
        Returns:
            frustum_features [torch.Tensor(N, C, D, H, W)]: Image features
        """
        channel_dim = 1
        depth_dim = 2

        # Resize to match dimensions
        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)

        # Apply softmax along depth axis and remove last depth category (> Max Range)
        depth_probs = F.softmax(depth_logits, dim=depth_dim)
        depth_probs = depth_probs[:, :, :-1]

        # Multiply to form image depth feature volume
        frustum_features = depth_probs * image_features
        return frustum_features


    def get_loss():
        return None
