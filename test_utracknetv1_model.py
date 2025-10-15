import torch
# 导入 models 包，这会触发 __init__.py 中的所有注册
import models_factory

# 1. 模拟一个配置文件 (未来可以从 yaml 文件加载)
cfg = dict(
    model=dict(
        type='UTrackNetV1',  # <-- 指定整车名字
        backbone=dict(
            type='UTrackNetV1Backbone',
            in_channels=9
        ),
        neck=dict(
            type='UTrackNetV1Neck'
        ),
        head=dict(
            type='UTrackNetV1Head',
            in_channels=64,
            out_channels=256
        )
    )
)

# 2. 从 models 包中直接调用总构建函数
# 注意，现在是从 models 直接导入，而不是 models.builder
model = models_factory.build_model(cfg['model'])

# 3. 打印模型，验证是否成功
print("--- 模型已通过新框架成功构建 ---")
print(model) # 取消注释可以查看详细结构

# 4. 进行一次完整的前向传播测试
mock_input = torch.randn(2, 9, 360, 640)
output = model(mock_input)

print(f"\n输入形状: {mock_input.shape}")
print(f"输出形状: {output.shape}")
assert output.shape == (2, 256, 360, 640)
print("\n✅ 框架测试通过！模型构建和前向传播均正常。")

# 想象一下未来...
# 如果你定义了一个 ResNetBackbone 并注册了它
# 你只需要修改配置文件的 'backbone': {'type': 'ResNetBackbone', ...}
# 无需改动任何其他代码！