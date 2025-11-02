import onnxruntime as ort
import numpy as np
import time

print("\n--- 正在测试 ONNX Runtime (GPU) ---")

# 1. 加载 ONNX 模型
# 确保 tracknetv5-r-str-fs.onnx 和 .data 文件在同一目录
onnx_path = 'tracknetv5-r-str-fs.onnx'
sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])

# 2. 准备输入数据
# (使用和 PyTorch 测试中相同的尺寸)
BATCH_SIZE = 1
INPUT_CHANNELS = 9
INPUT_HEIGHT = 288 # ??? 替换
INPUT_WIDTH = 512  # ??? 替换

# ONNX Runtime 使用 numpy 作为输入
dummy_input_np = np.random.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH).astype(np.float32)

# 获取 ONNX 模型的输入/输出名称
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# 3. 预热
print("正在预热...")
for _ in range(20):
    _ = sess.run([output_name], {input_name: dummy_input_np})

# 4. 正式计时
print("正在正式测试...")
num_iterations = 100
start_time = time.time()

for _ in range(num_iterations):
    _ = sess.run([output_name], {input_name: dummy_input_np})

end_time = time.time()

# 5. 计算结果
total_time = end_time - start_time
avg_latency_ms = (total_time / num_iterations) * 1000
print(f"ONNX Runtime (GPU) 平均延迟: {avg_latency_ms:.3f} ms")