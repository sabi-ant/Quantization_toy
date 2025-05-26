import torch
from model import CustomSeg
import edgeai_torchmodelopt
from datetime import datetime
import os
if __name__=="__main__":

    hhmmdd_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_dir = f"{hhmmdd_str}_torchmodelopt_r9.2"
    model = CustomSeg(num_classes=6, encoder_channels=[32, 64, 160, 400], pyramid_channels=32, neck_iter=3  )
    model = edgeai_torchmodelopt.xmodelopt.quantization.v2.QATFxModule(model, total_epochs=5)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001,weight_decay=0.0002)
    for i in range(5):
        img = torch.randn((1,3,352,480))
        seg_out, attr_out = model(img)
        attr_out = attr_out.permute(0,2,1)
        loss = torch.nn.functional.cross_entropy(seg_out, torch.randint(0, 6, (1, 88, 120)))
        attr_loss = torch.nn.functional.cross_entropy(attr_out, torch.randint(0, 14, (1,6)))
        loss += attr_loss
        loss.backward()
        optimizer.step()
    model.eval()
    model = model.convert()
    os.makedirs(dst_dir, exist_ok=True)
    dummy_input = torch.randn((1,3,352,480))
    torch.onnx.export(model, dummy_input, os.path.join(dst_dir,'model.onnx'),
                      input_names=['input'], output_names=['seg_out', 'attr_out'],
                       export_params=True, verbose=False, do_constant_folding=True, opset_version=17)


    print(seg_out.shape)
    print(attr_out.shape)
   