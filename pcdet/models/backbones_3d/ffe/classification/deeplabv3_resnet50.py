import torchvision

from .segmentation import Segmentation


class DeepLabV3_ResNet50(Segmentation):

    def __init__(self, **kwargs):
        """
        Initializes DeepLabV3 segmentation model
        """
        super().__init__(constructor=torchvision.models.segmentation.deeplabv3_resnet50, **kwargs)
