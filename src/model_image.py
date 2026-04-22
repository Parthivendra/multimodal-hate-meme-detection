import torch
import torch.nn as nn
from torchvision import models

class ImageModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ImageModel, self).__init__()

        self.backbone = models.resnet18(pretrained=True)

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, images):
        return self.backbone(images)
