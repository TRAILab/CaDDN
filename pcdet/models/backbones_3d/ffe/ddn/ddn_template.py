from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import kornia


class DDNTemplate(nn.Module):

    def __init__(self, constructor, feat_extract_layer, num_classes, pretrained_path=None, aux_loss=None):
        """
        Initializes depth distribution network.
        Args:
            constructor [function]: Model constructor
            feat_extract_layer [string]: Layer to extract features from
            pretrained_path [string]: (Optional) Path of the model to load weights from
            aux_loss [bool]: Flag to include auxillary loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path
        self.pretrained = pretrained_path is not None
        self.aux_loss = aux_loss

        if self.pretrained:
            # Preprocess Module
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])
            self.eps = 1e-4

        # Model
        self.model = self.get_model(constructor=constructor)
        self.feat_extract_layer = feat_extract_layer
        self.model.backbone.return_layers = {
            feat_extract_layer: 'features',
            **self.model.backbone.return_layers
        }

    def get_model(self, constructor):
        """
        Get model
        Args:
            constructor [function]: Model constructor
        Returns:
            model [nn.Module]: Model
        """
        # Get model
        model = constructor(pretrained=False,
                            pretrained_backbone=False,
                            num_classes=self.num_classes,
                            aux_loss=self.aux_loss)

        # Update weights
        if self.pretrained_path is not None:
            model_dict = model.state_dict()

            # Get pretrained state dict
            pretrained_dict = torch.load(self.pretrained_path)
            pretrained_dict = self.filter_pretrained_dict(model_dict=model_dict, pretrained_dict=pretrained_dict)

            # Update current model state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        return model

    def filter_pretrained_dict(self, model_dict, pretrained_dict):
        """
        Removes layers from pretrained state dict that are not used or changed in model
        Args:
            model_dict [dict]: Default model state dictionary
            pretrained_dict [dict]: Pretrained model state dictionary
        Returns:
            pretrained_dict [dict]: Pretrained model state dictionary with removed weights
        """
        # Removes aux classifier weights if not used
        if "aux_classifier.0.weight" in pretrained_dict and "aux_classifier.0.weight" not in model_dict:
            pretrained_dict = {key: value for key, value in pretrained_dict.items()
                               if "aux_classifier" not in key}

        # Removes final conv layer from weights if number of classes are different
        model_num_classes = model_dict["classifier.4.weight"].shape[0]
        pretrained_num_classes = pretrained_dict["classifier.4.weight"].shape[0]
        if model_num_classes != pretrained_num_classes:
            pretrained_dict.pop("classifier.4.weight")
            pretrained_dict.pop("classifier.4.bias")

        return pretrained_dict

    def forward(self, images):
        """
        Forward pass
        Args:
            images [torch.Tensor(N, 3, H_in, W_in)]: Input images
        Returns
            result [dict[torch.Tensor]]: Depth distribution result
                feat [torch.Tensor(N, C, H_out, W_out)]: Image features
                out [torch.Tensor(N, num_classes, H_out, W_out)]: Classification logits
                aux [torch.Tensor(N, num_classes, H_out, W_out)]: Auxillary classification scores
        """
        # Preprocess images
        x = self.preprocess(images)

        # Extract features
        result = OrderedDict()
        features = self.model.backbone(x)
        result['features'] = features['features']
        feat_shape = features['features'].shape[-2:]

        # Prediction classification logits
        x = features["out"]
        x = self.model.classifier(x)
        x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
        result["logits"] = x

        # Prediction auxillary classification logits
        if self.model.aux_classifier is not None:
            x = features["aux"]
            x = self.model.aux_classifier(x)
            x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images [torch.Tensor(N, 3, H, W)]: Input images
        Return
            x [torch.Tensor(N, 3, H, W)]: Preprocessed images
        """
        x = images
        if self.pretrained:
            # Create a mask for zeroing padded pixels (x < 0)
            mask = x < 0 - self.eps

            # Match ResNet pretrained preprocessing
            x = kornia.normalize(x, mean=self.norm_mean, std=self.norm_std)

            # Make padded pixels = 0
            x[mask] = 0

        return x
