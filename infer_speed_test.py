import torch
import time
from thop import profile
from models_factory.builder import build_model

# --- 1. “厨房重地”: 辅助函数和配置 ---
model_cfg = dict(
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
        in_channels=16,
        out_channels=3
    )
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("警告：没有检测到CUDA，测试将在CPU上进行，结果可能不准确。")

BATCH_SIZE = 2
CHANNELS = 13
HEIGHT = 288
WIDTH = 512

# --- 2. 计算模型参数、GFLOPs ---
# 【修改点】：我们先创建一个模型实例专门用于计算复杂度
print("\n--- 模型复杂度 ---")
try:
    # 建立模型
    model_for_flops = build_model(model_cfg).to(device)
    model_for_flops.eval()

    # 2.1 计算模型参数量 (Model Size)
    total_params = sum(p.numel() for p in model_for_flops.parameters())
    trainable_params = sum(p.numel() for p in model_for_flops.parameters() if p.requires_grad)
    print(f"总参数量 (Total Params): {total_params / 1e6:.2f} M")
    print(f"可训练参数 (Trainable Params): {trainable_params / 1e6:.2f} M")

    # 2.2 计算 GFLOPs (计算量)
    thop_input = torch.randn(1, CHANNELS, HEIGHT, WIDTH).to(device)
    flops, params = profile(model_for_flops, inputs=(thop_input, ), verbose=False)
    gflops = flops / 1e9
    print(f"单张图片 GFLOPs: {gflops:.2f} G")

    # 【修改点】：计算完成后，删除这个带钩子的模型
    del model_for_flops
    del thop_input
    torch.cuda.empty_cache() # 清理一下显存

except Exception as e:
    print(f"计算模型复杂度失败: {e}")
    exit()


# --- 3. 测量显存占用 和 速度测试 ---
# 【修改点】：在这里，我们重新创建一个“干净”的模型实例
print("\n--- 准备显存与速度测试 ---")
model = build_model(model_cfg)
model.to(device)
model.eval()

# 3.1 创建用于测试的模拟数据
try:
    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH,
                            dtype=torch.float32).to(device)
except RuntimeError as e:
    print(f"创建模拟数据时出错 (可能是显存不足): {e}")
    exit()

# 3.2 测量峰值显存占用
torch.cuda.synchronize(device)
torch.cuda.reset_peak_memory_stats(device)
try:
    with torch.no_grad():
        _ = model(dummy_input) # 运行一次
    torch.cuda.synchronize(device)
    
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"峰值显存占用 (Batch Size={BATCH_SIZE}): {peak_memory_mb:.2f} MB")
except RuntimeError as e:
    print(f"在测量显存时出错 (可能是OOM): {e}")
    exit()


# 3.3 速度测试
N_WARMUP = 50
N_RUNS = 200
print(f"开始速度测试... Batch Size: {BATCH_SIZE}, 预热: {N_WARMUP} 轮, 测试: {N_RUNS} 轮")

# 预热
with torch.no_grad():
    for _ in range(N_WARMUP):
        _ = model(dummy_input)
torch.cuda.synchronize(device)

# 计时测试
total_time = 0.0
with torch.no_grad():
    torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    for _ in range(N_RUNS):
        _ = model(dummy_input)

    torch.cuda.synchronize(device)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time

# --- 4. 计算结果 ---
total_images = BATCH_SIZE * N_RUNS
fps = total_images / total_time
batch_time_ms = (total_time / N_RUNS) * 1000

print("\n--- 速度测试结果 ---")
print(f"总耗时: {total_time:.4f} 秒")
print(f"平均 Batch 延迟: {batch_time_ms:.4f} 毫秒 (ms)")
print(f"纯模型吞吐量 (FPS): {fps:.2f} it/s (images/sec)")