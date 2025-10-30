
from pathlib import Path

# ------------------- 1. 模型定义 (Model) -------------------
model = dict(
    type='TrackNetV2LRMVDR',
    backbone=dict(
        type='TrackNetV2LRMVDRBackbone',
        in_channels=13
    ),
    neck=dict(
        type='TrackNetV2Neck'
    ),
    head=dict(
        type='TrackNetV2MVDRTSATTHead',
        in_channels=64,
        out_channels=3,
        IsDraft=True
    )
)

# ------------------- 2. 数据定义 (Data) -------------------
# --- 2.1 通用参数 ---
input_size = (288, 512)  # (height, width)
original_size = (720, 1280) # 原图片大小(height, width)
# ‼️ 请务必将此路径修改为您自己电脑上的正确路径
data_root = './data/v2'
output_dir = f'./work_dirs/{Path(__file__).stem}'
model_pth_path = './work_dirs/best_model.pth'

# --- 2.2 数据处理流水线定义 ---
pipeline = [
    dict(type='LoadMultiImagesFromPaths', to_rgb=True),
    dict(type='Resize', keys=['path_prev', 'path', 'path_next'], size=input_size),
    # dict(type='GenerateMotionAttention', threshold=40),
    dict(type='ConcatChannels',
         keys=['path_prev', 'path', 'path_next'],
         output_key='image'),
    dict(type='LoadAndFormatMultiTargets',  # 使用新的多目标加载器
         keys=['gt_path_prev', 'gt_path', 'gt_path_next'],  # 指定三个gt路径
         output_key='target'),
    dict(type='Finalize',
         image_key='image',
         final_keys=['image', 'target', 'coords', 'visibility', 'original_info'])
]

# --- 2.3 数据加载器配置 ---
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    val=dict(
        type='TrackNetV2Dataset',
        data_dir=data_root,
        csv_path=f'{data_root}/labels_context_val.csv',
        input_height=input_size[0],
        input_width=input_size[1],
        pipeline=pipeline
    )
)

seed = 42
resume_from = None