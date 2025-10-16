from .bifpn import BiFPN
import torch.nn as nn
from torchvision.models import regnet_x_400mf, RegNet_X_400MF_Weights
import torch

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
        self.neck = BiFPN(encoder_channels=[32, 64, 160, 400],pyramid_channels=pyramid_channels, num_layers=neck_iter, output_index=[0,2])
        self.maxpool = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conf = nn.Sequential(
            
            nn.Conv2d(pyramid_channels, pyramid_channels, 3, padding=1), 
            nn.BatchNorm2d(pyramid_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(pyramid_channels, 1, 1)
            )
        self.offset = nn.Sequential(
            
            nn.Conv2d(pyramid_channels, pyramid_channels, 3, padding=1),
            nn.BatchNorm2d(pyramid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(pyramid_channels, 2, 1)
        )
        self.embed = nn.Sequential(
            nn.Conv2d(pyramid_channels, pyramid_channels, 3, padding=1),
            nn.BatchNorm2d(pyramid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(pyramid_channels, 4, 1)
        )
        self.attr_pre = nn.Sequential(
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.Conv2d(4, 4, 3,1,1),
                                  nn.BatchNorm2d(4),
                                  nn.ReLU(inplace=True))
        self.dim_reduction = nn.Sequential(nn.Conv2d(4+32, 5, 1,1,0),
                                  nn.BatchNorm2d(5),
                                  nn.ReLU(inplace=True))
        self.attr = nn.Sequential(nn.Linear(16*32, 32),
                                  nn.LayerNorm(32),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(32, 8))
        self.load_pretrained_weight('/home/sabi/workspace/PINet_new/TuSimple/mlruns/0/eab9acf68889491c8d436dcdee717cac/artifacts/latest_model/latest_model.pth')
        
    def load_pretrained_weight(self, state_dict_path):
        """
        Load the state_dict into the model.
        """
        state_dict = torch.load(state_dict_path)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print(f"Fail to load state_dict from {state_dict_path} with strict=False")
            print(f"Missing keys in the state_dict: {missing_keys}")
            print(f"Unexpected keys in the state_dict: {unexpected_keys}")
        else:
            print(f"Successfully loaded state_dict from {state_dict_path}")
        

    def forward(self, x):
        features = self.backbone(x) # [p2, p3, p4, p5]
        features = self.neck(features) # [p2]
        rsz_feature = self.maxpool(features[0])
        conf =  self.conf(rsz_feature)
        offset = self.offset(rsz_feature)
        embed = self.embed(rsz_feature)
        attr_pre = self.attr_pre(embed)
        mixed_feat = torch.concat([attr_pre, features[1]], dim=1)
        attr_feat = self.dim_reduction(mixed_feat)
        attr_feat = attr_feat.flatten(2)
        attr_out = self.attr(attr_feat)
        return conf, offset, embed, attr_out

