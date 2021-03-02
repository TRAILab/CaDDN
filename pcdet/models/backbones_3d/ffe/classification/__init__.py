from .deeplabv3_resnet50 import DeepLabV3_ResNet50
from .deeplabv3_resnet101 import DeepLabV3_ResNet101

class_modules = {
    'DeepLabV3_ResNet50': DeepLabV3_ResNet50,
    'DeepLabV3_ResNet101': DeepLabV3_ResNet101,
}
