"""
This file has been modified by Cody Reading to remove the PartA2 Head
"""

from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .roi_head_template import RoIHeadTemplate

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PVRCNNHead': PVRCNNHead,
    'PointRCNNHead': PointRCNNHead
}
