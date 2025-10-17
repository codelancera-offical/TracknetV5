# ------------------- 1. 模型定义 (Model) -------------------
# 这部分内容将直接作为参数传递给 models/builder.py 中的 build_tracker 函数
model = dict(
    type='UTrackNetV1',  # 对应在 TRACKERS 注册表里注册的名字
    backbone=dict(
        type='UTrackNetBackbone', # 对应 BACKBONES 注册表里的名字
        in_channels=9  # 例如，输入是3个连续的RGB帧堆叠而成
    ),
    neck=dict(
        type='UTrackNetNeck' # 对应 NECKS 注册表里的名字
    ),
    head=dict(
        type='UTrackNetV1Head', # 对应 HEADS 注册表里的名字
        in_channels=64,
        out_channels=256 # 输出256个灰度等级
    )
)

# ------------------- 2. 数据定义 (Data) -------------------
# 定义数据处理的流水线 (pipeline)，即各种数据增强和预处理操作
train_pipeline = [
    dict(type='LoadImageFromFile'), # 从文件加载图像的示例
    dict(type='Resize', size=(640, 360)), # 调整图像大小
    dict(type='RandomFlip', prob=0.5), # 随机翻转
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]), # 归一化
    dict(type='ToTensor'), # 转换为Tensor
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(640, 360)),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    dict(type='ToTensor'),
]

# 将所有数据相关的配置整合到 data 字典中
data = dict(
    # 每个GPU处理的样本数 (Batch Size)
    samples_per_gpu=4,
    # DataLoader使用的工作线程数
    workers_per_gpu=4,
    
    # 训练集配置
    train=dict(
        type='SoccerNetDataset', # 假设的数据集类名
        data_root='/path/to/your/soccernet/train',
        ann_file='/path/to/your/soccernet/annotations/train.json',
        pipeline=train_pipeline
    ),
    # 验证集配置
    val=dict(
        type='SoccerNetDataset',
        data_root='/path/to/your/soccernet/val',
        ann_file='/path/to/your/soccernet/annotations/val.json',
        pipeline=val_pipeline
    ),
    # 测试集配置
    test=dict(
        type='SoccerNetDataset',
        data_root='/path/to/your/soccernet/test',
        ann_file='/path/to/your/soccernet/annotations/test.json',
        pipeline=val_pipeline
    )
)

# ------------------- 3. 优化策略 (Optimization) -------------------
# 定义优化器
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05)

# 定义学习率调度器 (LR Scheduler)
# 例如：一个在第150和200轮时降低学习率的策略
lr_config = dict(
    policy='step', # 策略类型
    step=[150, 200]  # 在哪些 epoch 降低学习率
)

# ------------------- 4. 运行时设置 (Runtime) -------------------
# 总训练轮数
total_epochs = 240

# 日志配置
log_config = dict(
    interval=50,  # 每 50 个 iteration 打印一次日志
    hooks=[
        dict(type='TextLoggerHook'), # 文本日志
        # dict(type='TensorboardLoggerHook') # Tensorboard 日志 (可选)
    ]
)

# 模型权重保存 (Checkpoint) 配置
checkpoint_config = dict(
    interval=10 # 每 10 个 epoch 保存一次模型权重
)

# 验证 (Evaluation) 配置
evaluation = dict(
    interval=10 # 每 10 个 epoch 在验证集上评估一次模型
)

# 其他运行时设置
seed = 42 # 随机种子，保证实验可复现
work_dir = f'./work_dirs/{__file__.split("/")[-1][:-3]}' # 将日志和模型保存在以本配置文件名命名的目录中
resume_from = None # 从某个checkpoint恢复训练