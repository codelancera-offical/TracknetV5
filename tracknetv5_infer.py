# -*- coding: utf-8 -*-
import torch
import cv2
import numpy as np
import argparse
import os
import math # 确保 math 被导入
import csv
from pathlib import Path
from tqdm import tqdm
from collections import deque

# 导入你项目里的构建器和模型！
from models_factory.builder import build_model
from datasets_factory.transforms.utracknetv1_transforms import (
    Resize, ConcatChannels
)

# --- 1. “厨房重地”: 辅助函数和配置 (这部分不变) ---
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
        type='R_STRHead',
        in_channels=64,
        out_channels=3 # <-- 你提到这现在是 3
    )
)


# --- ✨✨✨ 已修改的辅助函数 ✨✨✨ ---
def _heatmap_to_coords(heatmap: np.ndarray, threshold: int = 127, min_circularity: float = 0.7):
    """
    一个鲁棒的坐标提取函数。
    它对热力图进行二值化，然后寻找最大且符合圆度要求的轮廓的质心。
    """
    if heatmap.dtype != np.uint8:
        heatmap = heatmap.astype(np.uint8)

    _, binary_map = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            # 预先过滤掉非常小的噪点
            if area < 5: 
                continue
            
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
                
            # 计算圆度
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            
            if circularity >= min_circularity:
                valid_contours.append(c)

    if valid_contours:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy

    return None


def draw_comet_tail(frame, points_deque):
    """(此函数保持不变)"""
    overlay = np.zeros_like(frame, dtype=np.uint8)
    for i in range(1, len(points_deque)):
        if points_deque[i - 1] is None or points_deque[i] is None:
            continue
        alpha = i / len(points_deque)
        line_color = (0, 0, int(alpha * 255))
        pt1 = tuple(points_deque[i - 1])
        pt2 = tuple(points_deque[i])
        cv2.line(overlay, pt1, pt2, line_color, 2)
    frame = cv2.addWeighted(overlay, 1.0, frame, 1.0, 0)
    if points_deque and points_deque[-1] is not None:
        cv2.circle(frame, tuple(points_deque[-1]), 5, (0, 0, 255), -1)
    return frame

# --- 2. “核心加工车间”: ✨✨✨ 已重构的 process_video 函数 ✨✨✨ ---
def process_video(video_path: Path, model, device, args, output_root_dir: Path):
    """
    处理单个视频文件，并生成所有需要的输出文件。
    新逻辑：一次读取 3 帧，推理 3 帧，写入 3 帧，然后跳 3 帧。
    """
    print(f"\n🏭 Processing video: {video_path.name}")
    
    video_output_dir = output_root_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    input_size = (288, 512)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    trajectory_video_path = video_output_dir / f"{video_path.stem}_trajectory.mp4"
    comparison_video_path = video_output_dir / f"{video_path.stem}_comparison.mp4"
    csv_path = video_output_dir / f"{video_path.stem}_data.csv"
    
    writer_traj = cv2.VideoWriter(str(trajectory_video_path), fourcc, fps, (input_size[1], input_size[0]))
    writer_comp = cv2.VideoWriter(str(comparison_video_path), fourcc, fps, (input_size[1] * 2, input_size[0]))

    # 轨迹点保留不变，它只关心最近的 `fps` 个点
    trajectory_points = deque(maxlen=fps) 
    
    csv_data = []
    detected_frames_count = 0
    
    # 预处理转换（保持不变）
    resizer = Resize(keys=['path_prev', 'path', 'path_next'], size=input_size)
    concatenator = ConcatChannels(
        keys=['path_prev', 'path', 'path_next'],
        output_key='image'
    )
    
    # --- 新的循环逻辑 ---
    frame_idx_counter = 0
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.stem}")

    while cap.isOpened():
        # 1. 一次性读取 3 帧
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        ret3, frame3 = cap.read()

        # 如果任何一帧读取失败（视频末尾），则终止循环
        if not ret1 or not ret2 or not ret3:
            break

        # 2. 准备模型输入
        # (你提到模型内部处理，我们只需按转换器要求提供3帧)
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        frame3_rgb = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
        
        data_dict = {'path_prev': frame1_rgb, 'path': frame2_rgb, 'path_next': frame3_rgb}
        data_dict = resizer(data_dict)
        data_dict = concatenator(data_dict)
        
        # 存储调整大小后的帧，用于后续绘图
        # data_dict['path_prev'] 现在是调整后的 frame1
        resized_frames = [data_dict['path_prev'], data_dict['path'], data_dict['path_next']]
        
        image_np = data_dict['image']
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)

        # 3. 批量推理
        with torch.no_grad():
            # heatmap_preds 的形状是 [1, 3, H, W]
            heatmap_preds = model(image_tensor)
        
        # 移除 batch 维度，得到 (3, H, W) 的 NumPy 数组
        heatmaps_np = heatmap_preds.squeeze(0).cpu().numpy()
        threshold_uint8 = int(args.threshold * 255)

        # 4. 循环处理这 3 帧的结果
        for i in range(3):
            current_frame_idx = frame_idx_counter + i
            single_heatmap_np = heatmaps_np[i] # 形状 (H, W)
            heatmap_uint8 = (single_heatmap_np * 255).astype(np.uint8)

            # (A) 提取坐标
            coords = _heatmap_to_coords(heatmap_uint8, threshold=threshold_uint8, min_circularity=args.min_circularity)
            
            # (B) 记录 CSV 和轨迹
            if coords is not None:
                detected_frames_count += 1
                trajectory_points.append(coords)
                csv_row = {'frame_number': current_frame_idx, 'detected': 1, 'x': coords[0], 'y': coords[1]}
            else:
                trajectory_points.append(None)
                csv_row = {'frame_number': current_frame_idx, 'detected': 0, 'x': 0.0, 'y': 0.0}
            csv_data.append(csv_row)
            
            # (C) 绘制和写入视频
            frame_to_draw = cv2.cvtColor(resized_frames[i], cv2.COLOR_RGB2BGR)
            
            # 绘制轨迹视频
            final_traj_frame = draw_comet_tail(frame_to_draw, trajectory_points)
            writer_traj.write(final_traj_frame)

            # 绘制对比视频
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            combined_frame = np.hstack((final_traj_frame, heatmap_color))
            writer_comp.write(combined_frame)

        # 5. 更新计数器和进度条 (关键！)
        frame_idx_counter += 3
        pbar.update(3)
    
    # --- 循环结束后的清理 ---
    pbar.close() # 关闭进度条

    detection_ratio = (detected_frames_count / total_frames) if total_frames > 0 else 0
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame_number', 'detected', 'x', 'y'])
        writer.writeheader()
        writer.writerows(csv_data)
        f.write("\n")
        f.write(f"total_detected_frame,{detected_frames_count}\n")
        f.write(f"detection_ratio,{detection_ratio:.4f}\n")

    cap.release()
    writer_traj.release()
    writer_comp.release()
    print(f"✅ Finished processing. Results saved in: {video_output_dir}")

# --- 3. “总调度室”: main 函数 (保持不变) ---
def main():
    parser = argparse.ArgumentParser(description="TrackNetV5 Batch Inference Pipeline")
    parser.add_argument('input_dir', type=str, help='Path to the directory containing input videos.')
    parser.add_argument('weights_path', type=str, help='Path to the model weights (.pth file).')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for detection (0-1).')
    parser.add_argument('--min-circularity', type=float, default=0.7, help='Minimum circularity for a valid detection (0-1).')
    args = parser.parse_args()

    print("🚀 Starting Batch Inference Pipeline...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    model = build_model(model_cfg)
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu'))
    model.to(device).eval()
    print(f"✅ Model loaded from {args.weights_path} and sent to {device}.")

    input_dir = Path(args.input_dir)
    output_root_dir = input_dir / "utracknet_mvat_wbce"
    output_root_dir.mkdir(exist_ok=True)
    
    print("🔎 Searching for .mp4 and .mov files...")
    video_files = []
    supported_formats = ['*.mp4', '*.mov', '*.MOV', '*.MP4']
    for fmt in supported_formats:
        video_files.extend(input_dir.glob(fmt))
    
    if not video_files:
        print(f"❌ No supported video files (.mp4, .mov) found in {input_dir}. Exiting.")
        return
        
    video_files = sorted(list(set(video_files)))
    print(f"Found {len(video_files)} videos to process.")
    
    for video_path in video_files:
        process_video(video_path, model, device, args, output_root_dir)

    print(f"\n🎉🎉🎉 All videos processed! Check the results in: {output_root_dir} 🎉🎉🎉")

if __name__ == '__main__':
    main()