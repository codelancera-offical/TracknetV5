import torch
from batch_inference import model_cfg
from models_factory.builder import build_model
 
weight_path = r'C:\Users\lance\Desktop\TracknetV5\work_dirs\tracknetv5_r-str-fs-tennis_b2e500\epoch_5.pth'   # 原来模型保存的权重路径
onnx_net_path = './tracknetv5-r-str-fs.onnx'       # 设置onnx模型保存的权重路径
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

model_cfg = dict(
    type='TrackNetV5',
    backbone=dict(
        type='TrackNetV2Backbone', # OK
        in_channels=13
    ),
    neck=dict(
        type='TrackNetV2Neck'# OK
    ),
    head=dict( 
        type='R_STRHeadFS',
        in_channels=64,
        out_channels=3
    )
)
 
# 权重导入模型
model = build_model(model_cfg)
model.load_state_dict(torch.load(weight_path, map_location='cpu'))
model.to(device).eval()
 
input = torch.randn(1, 9, 288, 512).to(device)   # (B,C,H,W)  其中Batch必须为1，因为test时一般为1，尺寸 H,W 必须和训练时的尺寸一致
torch.onnx.export(model, input, onnx_net_path, verbose=False)
