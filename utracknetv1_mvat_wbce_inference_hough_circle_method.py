# -*- coding: utf-8 -*-
import torch
import cv2
import numpy as np
import argparse
import os
import math
import csv 
from pathlib import Path
from tqdm import tqdm
from collections import deque

# 导入你项目里的构建器和模型！
from models_factory.builder import build_model
from datasets_factory.transforms.utracknetv1_transforms import (
    Resize, GenerateMotionAttention, ConcatChannels
)

# --- 1. “厨房重地”: 辅助函数和配置 ---
model_cfg = dict(
    type='UTrackNetV1',
    backbone=dict(
        type='UTrackNetV1Backbone',
        in_channels=13
    ),
    neck=dict(
        type='UTrackNetV1Neck'
    ),
    head=dict(
        type='UTrackNetV1HeadSigmoid',
        in_channels=64,
        out_channels=1
    )
)

def _heatmap_to_coords(heatmap: np.ndarray, binary_threshold: int = 50, circle_threshold: int = 5, min_radius: int = 2, max_radius: int = 15):
    """
    使用霍夫圆变换将热力图转换为坐标点和半径。
    """
    if heatmap.dtype != np.uint8:
        heatmap_uint8 = heatmap.astype(np.uint8)
    else:
        heatmap_uint8 = heatmap
    
    _, binary_map = cv2.threshold(heatmap_uint8, binary_threshold, 255, cv2.THRESH_BINARY)
    
    circles = cv2.HoughCircles(
        image=binary_map,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=40,
        param1=50,
        param2=circle_threshold,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None:
        x, y, r = circles[0][0]
        return int(x), int(y), r
    
    return None, None, None

def draw_comet_tail(frame, points_deque):
    """
    绘制带有中断效果的彗星尾巴轨迹。
    """
    overlay = np.zeros_like(frame, dtype=np.uint8)
    for i in range(1, len(points_deque)):
        # 关键！如果当前点或前一个点是None (即轨迹中断)，则跳过绘制这一段线
        if points_deque[i - 1] is None or points_deque[i] is None:
            continue
        
        alpha = i / len(points_deque)
        line_color = (0, 0, int(alpha * 255)) # 尾巴是蓝色渐变
        pt1 = tuple(points_deque[i - 1])
        pt2 = tuple(points_deque[i])
        cv2.line(overlay, pt1, pt2, line_color, 2)
        
    frame = cv2.addWeighted(overlay, 1.0, frame, 1.0, 0)
    
    # 只在当前帧检测到球时，才绘制红色的“头”
    if points_deque and points_deque[-1] is not None:
        cv2.circle(frame, tuple(points_deque[-1]), 5, (0, 0, 255), -1) # 红色的头
        
    return frame

# --- 2. “核心加工车间”: 处理单个视频的函数 ---
def process_video(video_path: Path, model, device, args, output_root_dir: Path):
    print(f"\n🏭 Processing video: {video_path.name}")
    
    video_output_dir = output_root_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    input_size = (360, 640)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    trajectory_video_path = video_output_dir / f"{video_path.stem}_trajectory.mp4"
    comparison_video_path = video_output_dir / f"{video_path.stem}_comparison.mp4"
    csv_path = video_output_dir / f"{video_path.stem}_data.csv"
    
    writer_traj = cv2.VideoWriter(str(trajectory_video_path), fourcc, fps, (input_size[1], input_size[0]))
    writer_comp = cv2.VideoWriter(str(comparison_video_path), fourcc, fps, (input_size[1] * 2, input_size[0]))

    frame_buffer = deque(maxlen=3)
    trajectory_points = deque(maxlen=fps) 
    
    csv_data = []
    detected_frames_count = 0
    detected_radii = [] 
    
    resizer = Resize(keys=['path_prev', 'path', 'path_next'], size=input_size)
    motion_generator = GenerateMotionAttention(threshold=40)
    concatenator = ConcatChannels(
        keys=['path_prev', 'att_prev_to_curr', 'path', 'att_curr_to_next', 'path_next'],
        output_key='image'
    )
    
    for frame_idx in tqdm(range(total_frames), desc=f"Processing {video_path.stem}"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(frame_rgb)
        
        # --- ✅ 修复点：在这里初始化变量 ---
        cx, cy, radius = None, None, None
        
        heatmap_uint8 = np.zeros(input_size, dtype=np.uint8)
        if len(frame_buffer) == 3:
            prev, curr, next_ = frame_buffer[0], frame_buffer[1], frame_buffer[2]
            
            data_dict = {'path_prev': prev, 'path': curr, 'path_next': next_}
            data_dict = resizer(data_dict)
            data_dict = motion_generator(data_dict)
            data_dict = concatenator(data_dict)
            
            image_np = data_dict['image']
            image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)

            with torch.no_grad():
                heatmap_pred = model(image_tensor).squeeze().cpu().numpy()
            
            heatmap_uint8 = (heatmap_pred * 255).astype(np.uint8)
            threshold_uint8 = int(args.threshold * 255)
            
            cx, cy, radius = _heatmap_to_coords(
                heatmap_uint8,
                binary_threshold=threshold_uint8,
                circle_threshold=args.circle_threshold,
                min_radius=args.min_radius,
                max_radius=args.max_radius
            )
        
        if cx is not None:
            detected_frames_count += 1
            detected_radii.append(radius)
            coords = (cx, cy)
            trajectory_points.append(coords)
            csv_row = {'frame_number': frame_idx, 'detected': 1, 'x': cx, 'y': cy, 'radius': radius}
        else:
            trajectory_points.append(None)
            csv_row = {'frame_number': frame_idx, 'detected': 0, 'x': 0.0, 'y': 0.0, 'radius': 0.0}
        csv_data.append(csv_row)
        
        if len(frame_buffer) >= 2:
            frame_to_process = frame_buffer[1]
            frame_to_draw = cv2.resize(frame_to_process, (input_size[1], input_size[0]))
            frame_to_draw = cv2.cvtColor(frame_to_draw, cv2.COLOR_RGB2BGR)
            final_traj_frame = draw_comet_tail(frame_to_draw, trajectory_points)
            writer_traj.write(final_traj_frame)

            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            combined_frame = np.hstack((final_traj_frame, heatmap_color))
            writer_comp.write(combined_frame)

    detection_ratio = (detected_frames_count / total_frames) if total_frames > 0 else 0
    average_radius = np.mean(detected_radii) if detected_radii else 0.0

    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['frame_number', 'detected', 'x', 'y', 'radius']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
        f.write("\n")
        f.write(f"total_detected_frame,{detected_frames_count}\n")
        f.write(f"detection_ratio,{detection_ratio:.4f}\n")
        f.write(f"average_radius,{average_radius:.4f}\n")

    cap.release()
    writer_traj.release()
    writer_comp.release()
    
    print(f"✅ Finished processing. Average detected radius: {average_radius:.2f} px.")
    print(f"   Results saved in: {video_output_dir}")
    
# --- 3. “总调度室”: main 函数 ---
def main():
    parser = argparse.ArgumentParser(description="TrackNetV5 Batch Inference Pipeline")
    parser.add_argument('input_dir', type=str, help='Path to the directory containing input videos.')
    parser.add_argument('weights_path', type=str, help='Path to the model weights (.pth file).')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--threshold', type=float, default=0.2, help='Binary threshold for heatmap (0-1).')
    parser.add_argument('--circle-threshold', type=int, default=5, help='Hough Circle accumulator threshold (e.g., 2-10). Lower is more lenient.')
    parser.add_argument('--min-radius', type=int, default=2, help='Minimum radius of the ball to detect.')
    parser.add_argument('--max-radius', type=int, default=15, help='Maximum radius of the ball to detect.')
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