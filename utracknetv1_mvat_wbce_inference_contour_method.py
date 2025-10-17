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

# --- 1. “厨房重地”: 辅助函数和配置 (这部分不变) ---
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


def _heatmap_to_coords(heatmap: np.ndarray, threshold: int = 80, min_circularity: float = 0.7):
    """
    一个鲁棒的坐标提取函数。
    它对热力图进行二值化，然后寻找最大轮廓的质心作为坐标。
    """
    if heatmap.dtype != np.uint8:
        heatmap = heatmap.astype(np.uint8)
    _, binary_map = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter == 0:
            # ✨✨✨ 修正点 1: 失败时，返回一个唯一的、清晰的 None ✨✨✨
            return None
        circularity = (4 * math.pi * area) / (perimeter * perimeter)
        if circularity >= min_circularity:
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # ✨✨✨ 修正点 1: 成功时返回一个清晰的元组 ✨✨✨
                return (cx, cy)
    # ✨✨✨ 修正点 1: 任何失败路径都最终返回唯一的 None ✨✨✨
    return None


def draw_comet_tail(frame, points_deque):
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

# --- 2. “核心加工车间”: 处理单个视频的函数 ---
def process_video(video_path: Path, model, device, args, output_root_dir: Path):
    """
    处理单个视频文件，并生成所有需要的输出文件。
    """
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
        
        coords = None
        heatmap_uint8 = np.zeros(input_size, dtype=np.uint8) # 预先定义，避免引用错误
        
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
            coords = _heatmap_to_coords(heatmap_uint8, threshold=threshold_uint8, min_circularity=args.min_circularity)
        
        # --- ✨✨✨ 终极修正：用 'is not None' 来进行精确判断 ✨✨✨
        if coords is not None:
            detected_frames_count += 1
            trajectory_points.append(coords)
            csv_row = {'frame_number': frame_idx, 'detected': 1, 'x': coords[0], 'y': coords[1]}
        else:
            trajectory_points.append(None)
            csv_row = {'frame_number': frame_idx, 'detected': 0, 'x': 0.0, 'y': 0.0}
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

# --- 3. “总调度室”: main 函数 ---
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