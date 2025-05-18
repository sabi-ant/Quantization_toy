from bifpn import BiFPN
import torch.nn as nn
from torchvision.models import regnet_x_400mf, RegNet_X_400MF_Weights

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        weights = RegNet_X_400MF_Weights.DEFAULT
        tmp_model = regnet_x_400mf(weights=weights)
        self.stem = tmp_model.stem
        self.stages = nn.ModuleList(tmp_model.trunk_output)

    def forward(self, x):
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

class CustomSeg(nn.Module):
    def __init__(self, num_classes=1, encoder_channels=[32, 64, 160, 400], pyramid_channels=32, neck_iter=3):
        super().__init__()
               
        self.backbone = Backbone()
        self.neck = BiFPN(encoder_channels=[32, 64, 160, 400],pyramid_channels=pyramid_channels, num_layers=neck_iter)
        self.head = nn.Sequential(nn.Conv2d(pyramid_channels, pyramid_channels, 3, padding=1), 
                                  nn.BatchNorm2d(pyramid_channels), 
                                  nn.ReLU(inplace=True), 
                                  nn.Conv2d(pyramid_channels, num_classes, 1))

    def forward(self, x):
        features = self.backbone(x) # [p2, p3, p4, p5]
        features = self.neck(features) # [p2]
        return self.head(features[0])

