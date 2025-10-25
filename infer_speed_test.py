import torch
import time

# 导入你项目里的构建器和模型！
from models_factory.builder import build_model


# --- 1. “厨房重地”: 辅助函数和配置 ---
model_cfg = dict(
    type='TrackNetV1',
    backbone=dict(
        type='TrackNetV1Backbone',
    ),
    neck=dict(
        type='TrackNetV1Neck'
    ),
    head=dict(
        type='TrackNetV1Head',
    )
)

model = build_model(model_cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("警告：没有检测到CUDA，测试将在CPU上进行，结果可能不准确。")

model.eval()

BATCH_SIZE = 32
CHANNELS = 9
HEIGHT = 360
WIDTH = 640

try:
    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH, 
                            dtype=torch.float32).to(device)
except RuntimeError as e:
    print(f"创建模拟数据时出错 (可能是显存不足): {e}")
    # 如果显存不足，你可能需要减小 BATCH_SIZE
    exit()

# 定义测试的迭代次数
N_WARMUP = 50  # 预热迭代次数
N_RUNS = 200   # 实际测试迭代次数

print(f"开始测试... Batch Size: {BATCH_SIZE}, 预热: {N_WARMUP} 轮, 测试: {N_RUNS} 轮")

# --- 3. 预热 (Warm-up) ---
# 第一次运行CUDA操作时，它需要一些时间来初始化。
# 我们运行几轮 "预热" 迭代，确保测量的不是这些初始化的开销。
with torch.no_grad():
    for _ in range(N_WARMUP):
        _ = model(dummy_input)

# 确保预热操作已在GPU上完成
torch.cuda.synchronize(device)

# --- 4. 开始计时测试 ---
total_time = 0.0
with torch.no_grad():
    # 强制GPU在开始计时前完成所有待处理任务
    torch.cuda.synchronize(device)
    
    # 使用 time.perf_counter() 进行高精度计时
    start_time = time.perf_counter()

    for _ in range(N_RUNS):
        _ = model(dummy_input)

    # 强制GPU在结束计时前完成所有 N_RUNS 轮的推理
    torch.cuda.synchronize(device)
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time

# --- 5. 计算结果 ---

# N_RUNS 次迭代处理的总图像数
total_images = BATCH_SIZE * N_RUNS

# 计算平均每秒处理的图像数 (FPS)
fps = total_images / total_time

# 计算平均处理一个 batch 需要的时间 (ms)
batch_time_ms = (total_time / N_RUNS) * 1000

print("\n--- 测试结果 ---")
print(f"总耗时: {total_time:.4f} 秒")
print(f"平均 Batch 延迟: {batch_time_ms:.4f} 毫秒 (ms)")
print(f"纯模型吞吐量 (FPS): {fps:.2f} it/s (images/sec)")