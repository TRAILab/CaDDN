"""
This file has been modified by Cody Reading to remove support for spconv based modules
"""

from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG

__all__ = {
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
}
