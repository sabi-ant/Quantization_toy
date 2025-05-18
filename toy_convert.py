import torch
from model import CustomSeg

if __name__=="__main__":
    model = CustomSeg(num_classes=6, encoder_channels=[32, 64, 160, 400], pyramid_channels=32, neck_iter=3  )
    img = torch.randn((1,3,352,480))
    out = model(img)
    print(out.shape)