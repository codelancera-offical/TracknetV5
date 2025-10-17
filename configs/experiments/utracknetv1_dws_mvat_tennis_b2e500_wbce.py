
from pathlib import Path

# ------------------- 1. 模型定义 (Model) -------------------
model = dict(
    type='UTrackNetV1',
    backbone=dict(
        type='UTrackNetV1DWSBackbone',
        # Attention流水线输出13个通道
        in_channels=13
    ),
    neck=dict(
        type='UTrackNetV1DWSNeck'
    ),
    head=dict(
        type='UTrackNetV1DWSHeadSigmoid',
        in_channels=64,
        out_channels=1
    )
)

# ------------------- 2. 数据定义 (Data) -------------------
# --- 2.1 通用参数 ---
input_size = (360, 640)  # (height, width)
# ‼️ 请务必将此路径修改为您自己电脑上的正确路径
data_root = './data/tracknet'

# --- 2.2 数据处理流水线定义 ---
attention_pipeline = [
    dict(type='LoadMultiImagesFromPaths', to_rgb=True),
    dict(type='Resize', keys=['path_prev', 'path', 'path_next'], size=input_size),
    dict(type='GenerateMotionAttention', threshold=40),
    dict(type='ConcatChannels',
         keys=['path_prev', 'att_prev_to_curr', 'path', 'att_curr_to_next', 'path_next'],
         output_key='image'),
    dict(type='LoadAndFormatTarget'),
    dict(type='Finalize',
         image_key='image',
         final_keys=['image', 'target', 'coords', 'visibility', 'original_info'])
]

# --- 2.3 数据加载器配置 ---
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='TrackNetDataset',
        data_dir=data_root,
        csv_path=f'{data_root}/labels_context_train.csv',
        input_height=input_size[0],
        input_width=input_size[1],
        pipeline=attention_pipeline
    ),
    val=dict(
        type='TrackNetDataset',
        data_dir=data_root,
        csv_path=f'{data_root}/labels_context_val.csv',
        input_height=input_size[0],
        input_width=input_size[1],
        pipeline=attention_pipeline
    )
)

# ------------------- 3. 损失函数定义 (Loss) -------------------

loss = dict(
    type='UTrackNetV2LossWith1Channel'
)

# ------------------- 4. 优化策略定义 (Optimization) -------------------
# ✨ 修正一：根据您的要求，更换为 Adadelta 优化器，学习率为 1.0
optimizer = dict(type='Adadelta', lr=1.0)

# ✨ 修正二：根据您的要求，移除了学习率调度器 (lr_config)

# ------------------- 5. 评估策略定义 (Evaluation) -------------------
evaluation = dict(
    interval=100,
    metric=dict(
        type='UTrackNetV1Metric',
        min_dist=10,
    )
)

# ------------------- 6. 运行时定义 (Runtime) -------------------
total_epochs = 500
work_dir = f'./work_dirs/{Path(__file__).stem}'

# ✨ 修正三：根据您的要求，添加每轮最大迭代次数
steps_per_epoch = 200

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

custom_hooks = [
    dict(type='ValidationVisualizerHookWBCE', num_samples_to_save=10)
]

seed = 42
resume_from = None