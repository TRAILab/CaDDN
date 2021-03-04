import torch
import torch.nn as nn

from .frustum_grid_generator import FrustumGridGenerator
from .sampler import Sampler


class FrustumToVoxel(nn.Module):

    def __init__(self, model_cfg, grid_size, pc_range, disc_cfg):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            model_cfg [EasyDict]: Module configuration
            grid_size [np.array(3)]: Voxel grid shape [X, Y, Z]
            pc_range [list]: Voxelization point cloud range [X_min, Y_min, Z_min, X_max, Y_max, Z_max]
            disc_cfg [int]: Depth discretiziation configuration
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.disc_cfg = disc_cfg
        self.grid_generator = FrustumGridGenerator(grid_size=grid_size,
                                                   pc_range=pc_range,
                                                   disc_cfg=disc_cfg)
        self.sampler = Sampler(**model_cfg.SAMPLER)

    def forward(self, batch_dict, **kwargs):
        """
        Generates voxel features via 3D transformation and sampling
        Args:
            frustum_features [torch.Tensor(B, C, D, H_image, W_image)]: Image frustum features
            lidar_to_cam [torch.Tensor(B, 4, 4)]: LiDAR to camera frame transformation
            cam_to_img [torch.Tensor(B, 3, 4)]: Camera projection matrix
            image_shape [torch.Tensor(B, 2)]: Image shape [H, W]
        Returns:
            voxel_features [torch.Tensor(B, C, Z, Y, X)]: Image voxel features
        """
        # Generate sampling grid for frustum volume
        grid = self.grid_generator(lidar_to_cam=lidar_to_cam,
                                   cam_to_img=cam_to_img,
                                   image_shape=image_shape)  # (B, X, Y, Z, 3)

        # Sample frustum volume to generate voxel volume
        voxel_features = self.sampler(input_features=frustum_features,
                                      grid=grid)  # (B, C, X, Y, Z)

        # (B, C, X, Y, Z) -> (B, C, Z, Y, X)
        voxel_features = voxel_features.permute(0, 1, 4, 3, 2)
        return batch_dict
