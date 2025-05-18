"""
Implementation of Bidirectional Feature Pyramid Network module (BiFPN)
Reference: EfficientDet: Scalable and Efficient Object Detection - https://arxiv.org/abs/1911.09070
This version supports any number of input feature maps and is suitable for segmentation as well
hacked by @bonlime
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiFPNLastLayer(nn.Module):
    def __init__(self, num_features=5, channels=64):
        super().__init__()
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up5 = nn.Upsample(scale_factor=2, mode="nearest")

        self.up_norm2 = nn.BatchNorm2d(channels)
        self.up_norm3 = nn.BatchNorm2d(channels)
        self.up_norm4 = nn.BatchNorm2d(channels)

        self.up_act2 = nn.ReLU(inplace=True)
        self.up_act3 = nn.ReLU(inplace=True)
        self.up_act4 = nn.ReLU(inplace=True)


        self.up_conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.up_conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.up_conv4 = nn.Conv2d(channels, channels, 3, padding=1)
        self.up_conv5 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, features):
        p2_in, p3_in, p4_in, p5_in = features
        up_4 = self.up5(p5_in)
        up_4 = self.up_conv4(self.up_norm4(self.up_act4(p4_in + up_4)))
        up_3 = self.up4(up_4)
        up_3 = self.up_conv3(self.up_norm3(self.up_act3(p3_in + up_3)))
        up_2 = self.up2(up_3)
        up_2 = self.up_conv2(self.up_norm2(self.up_act2(p2_in + up_2)))

        return [up_2, up_3, up_4, p5_in]

class BiFPNLayer(nn.Module):
    """Builds one layer of Bi-directional Feature Pyramid Network
    Args:
        channels (int): Number of channels in each feature map after BiFPN. Defaults to 64.

    Input:
        features (List): 5 feature maps from encoder with resolution from 1/128 to 1/8

    Returns:
        p_out: features processed by 1 layer of BiFPN
    """

    def __init__(self, num_features=5, channels=64):
        super().__init__()

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up5 = nn.Upsample(scale_factor=2, mode="nearest")

        self.up_conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.up_conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.up_conv4 = nn.Conv2d(channels, channels, 3, padding=1)
        self.up_conv5 = nn.Conv2d(channels, channels, 3, padding=1)

        self.up_norm2 = nn.BatchNorm2d(channels)
        self.up_norm3 = nn.BatchNorm2d(channels)
        self.up_norm4 = nn.BatchNorm2d(channels)

        self.up_act2 = nn.ReLU(inplace=True)
        self.up_act3 = nn.ReLU(inplace=True)
        self.up_act4 = nn.ReLU(inplace=True)

        self.down2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.down3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.down4 = nn.MaxPool2d(3, stride=2, padding=1)
        self.down5 = nn.MaxPool2d(3, stride=2, padding=1)

        self.down_act3 = nn.ReLU(inplace=True)
        self.down_act4 = nn.ReLU(inplace=True)
        self.down_act5 = nn.ReLU(inplace=True)

        self.down_norm3 = nn.BatchNorm2d(channels)
        self.down_norm4 = nn.BatchNorm2d(channels)
        self.down_norm5 = nn.BatchNorm2d(channels)

        self.down_conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.down_conv4 = nn.Conv2d(channels, channels, 3, padding=1)
        self.down_conv5 = nn.Conv2d(channels, channels, 3, padding=1)


    def forward(self, features):
        p2_in, p3_in, p4_in, p5_in = features
        up_4 = self.up5(p5_in)
        up_4 = self.up_conv4(self.up_norm4(self.up_act4(p4_in + up_4)))
        up_3 = self.up4(up_4)
        up_3 = self.up_conv3(self.up_norm3(self.up_act3(p3_in + up_3)))
        up_2 = self.up2(up_3)
        up_2 = self.up_conv2(self.up_norm2(self.up_act2(p2_in + up_2)))

        down_3 = self.down2(up_2)
        down_3 = self.down_conv3(self.down_norm3(self.down_act3(down_3 + up_3)))
        down_4 = self.down3(down_3)
        down_4 = self.down_conv4(self.down_norm4(self.down_act4(down_4 + up_4)))
        down_5 = self.down4(down_4)
        down_5 = self.down_conv5(self.down_norm5(self.down_act5(down_5 + p5_in)))
        return [up_2, down_3, down_4, down_5]
        

class BiFPN(nn.Module):
    """
    Implementation of Bi-directional Feature Pyramid Network

    Args:
        encoder_channels (List[int]): Number of channels for each feature map from low res to high res.
        pyramid_channels (int): Number of channels in each feature map after BiFPN. Defaults to 64.
        num_layers (int): Number or repeats for BiFPN block. Default is 2

    Input:
        features (List): 5 feature maps from encoder [low_res, ... , high_res]

    https://arxiv.org/pdf/1911.09070.pdf
    """

    def __init__(self, encoder_channels, pyramid_channels=32, num_layers=2):
        super().__init__()
        # First layer preprocesses raw encoder features
        self.lateral_convs=nn.ModuleList(
            nn.Conv2d(c, pyramid_channels, 1) for c in encoder_channels)
        # Apply BiFPN block `num_layers` times
        bifpns=[]
        for _ in range(num_layers - 1):
            bifpns.append(BiFPNLayer(len(encoder_channels), pyramid_channels))
        bifpns.append(BiFPNLastLayer(len(encoder_channels), pyramid_channels))
        self.bifpns = nn.ModuleList(bifpns)
    
    def forward(self, features):
        features = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        for bifpn in self.bifpns:
            features = bifpn(features)
        return [features[0]]
        