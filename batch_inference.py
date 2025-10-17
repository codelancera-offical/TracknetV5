# -*- coding: utf-8 -*-
"""
TrackNetV5 Batch Inference Pipeline (V7 - Ultimate Chunked Parallel Edition)
Author: David (Your AI Software Engineer) & You!
Date: 2025-10-16
Description:
    The definitive version. This script uses a chunked, multi-process
    producer-consumer architecture to eliminate IPC overhead. Chunks of frames
    are distributed to worker processes, maximizing CPU utilization for
    pre-processing and keeping the GPU pipeline fully saturated.
    This is the final form. This is true speed.
"""
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
import multiprocessing as mp
import time

# --- 导入你项目里的构建器和模型 ---
from models_factory.builder import build_model
from datasets_factory.transforms.utracknetv1_transforms import (
    Resize, GenerateMotionAttention, ConcatChannels
)

# --- 1. 辅助函数和模型配置 (不变，已折叠) ---
model_cfg = dict(type='UTrackNetV1', backbone=dict(type='UTrackNetV1Backbone', in_channels=13), neck=dict(type='UTrackNetV1Neck'), head=dict(type='UTrackNetV1HeadSigmoid', in_channels=64, out_channels=1))
def _heatmap_to_coords(heatmap: np.ndarray, threshold: int = 50, min_circularity: float = 0.7):
    if heatmap.dtype != np.uint8: heatmap = heatmap.astype(np.uint8)
    _, binary_map = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter == 0: return None
        circularity = (4 * math.pi * area) / (perimeter * perimeter)
        if circularity >= min_circularity:
            M = cv2.moments(largest_contour)
            if M["m00"] > 0: return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None
def draw_comet_tail(frame, points_deque):
    overlay = np.zeros_like(frame, dtype=np.uint8)
    for i in range(1, len(points_deque)):
        if points_deque[i - 1] is None or points_deque[i] is None: continue
        alpha = i / len(points_deque)
        line_color = (0, 0, int(alpha * 255))
        pt1, pt2 = tuple(points_deque[i - 1]), tuple(points_deque[i])
        cv2.line(overlay, pt1, pt2, line_color, 2)
    frame = cv2.addWeighted(overlay, 1.0, frame, 1.0, 0)
    if points_deque and points_deque[-1] is not None: cv2.circle(frame, tuple(points_deque[-1]), 5, (0, 0, 255), -1)
    return frame

# --- ✨✨✨ 2. "帮厨"的工作指南 (处理大宗货物版) ✨✨✨ ---
def worker_preprocess_chunk(task_chunk):
    """
    每个CPU "帮厨" 现在接收一大块任务 (a chunk)。
    他在自己的厨房里处理完所有任务，然后一次性返回所有结果。
    """
    results_chunk = []
    # 帮厨们只需要初始化一次工具
    resizer = Resize(keys=['path_prev', 'path', 'path_next'], size=(360, 640))
    motion_generator = GenerateMotionAttention(threshold=40)
    concatenator = ConcatChannels(
        keys=['path_prev', 'att_prev_to_curr', 'path', 'att_curr_to_next', 'path_next'],
        output_key='image'
    )
    
    for task in task_chunk:
        frame_idx, prev, curr, next_ = task

        data_dict = {'path_prev': prev, 'path': curr, 'path_next': next_}
        data_dict = resizer(data_dict)
        data_dict = motion_generator(data_dict)
        data_dict = concatenator(data_dict)
        
        image_np = data_dict['image']
        # 注意：这里我们返回numpy数组，让主进程来转Tensor，可以避免一些序列化问题
        
        frame_for_drawing = cv2.resize(curr, (640, 360))
        frame_for_drawing_bgr = cv2.cvtColor(frame_for_drawing, cv2.COLOR_RGB2BGR)

        results_chunk.append((frame_idx, image_np, frame_for_drawing_bgr))
        
    return results_chunk

# --- ✨✨✨ 3. "流水线传送带" (大货箱版) ✨✨✨ ---
def frame_chunk_generator(video_path, total_frames, chunk_size):
    """
    这个生成器不再一次产出一个任务，而是一次产出一大箱任务 (a chunk)。
    """
    cap = cv2.VideoCapture(str(video_path))
    frame_buffer = deque(maxlen=3)
    
    tasks_chunk = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(frame_rgb)

        if len(frame_buffer) == 3:
            tasks_chunk.append((i - 1, frame_buffer[0], frame_buffer[1], frame_buffer[2]))
            if len(tasks_chunk) == chunk_size:
                yield tasks_chunk
                tasks_chunk = []
    
    if tasks_chunk: # 不要忘了最后一箱不满的货物
        yield tasks_chunk
            
    cap.release()

# --- 4. 核心加工车间 (最终形态) ---
def process_video(video_path: Path, model, device, args, output_root_dir: Path):
    print(f"\n🏭 Processing '{video_path.name}' with {args.num_workers} workers, batch_size={args.batch_size}, chunk_size={args.chunk_size}")
    
    # ... (文件和视频写入器设置不变) ...
    video_output_dir = output_root_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    input_size = (360, 640)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    trajectory_video_path = video_output_dir / f"{video_path.stem}_trajectory.mp4"
    comparison_video_path = video_output_dir / f"{video_path.stem}_comparison.mp4"
    csv_path = video_output_dir / f"{video_path.stem}_data.csv"
    writer_traj = cv2.VideoWriter(str(trajectory_video_path), fourcc, fps, (input_size[1], input_size[0]))
    writer_comp = cv2.VideoWriter(str(comparison_video_path), fourcc, fps, (input_size[1] * 2, input_size[0]))

    trajectory_points = deque(maxlen=fps)
    csv_data = []
    detected_frames_count = 0
    
    def _process_gpu_batch(tensors_to_process, frames_to_process, frame_indices):
        # 这个内部函数和之前几乎一样，只是现在由主进程全权负责
        nonlocal detected_frames_count
        if not tensors_to_process: return

        input_batch = torch.cat(tensors_to_process, dim=0).to(device)
        with torch.no_grad():
            heatmaps_batch = model(input_batch).squeeze(1).cpu().numpy()
        
        for i, heatmap_pred in enumerate(heatmaps_batch):
            frame_to_draw_bgr, frame_idx = frames_to_process[i], frame_indices[i]
            heatmap_uint8 = (heatmap_pred * 255).astype(np.uint8)
            threshold_uint8 = int(args.threshold * 255)
            coords = _heatmap_to_coords(heatmap_uint8, threshold=threshold_uint8, min_circularity=args.min_circularity)
            
            if coords:
                detected_frames_count += 1; trajectory_points.append(coords)
                csv_row = {'frame_number': frame_idx, 'detected': 1, 'x': coords[0], 'y': coords[1]}
            else:
                trajectory_points.append(None)
                csv_row = {'frame_number': frame_idx, 'detected': 0, 'x': 0.0, 'y': 0.0}
            csv_data.append(csv_row)
            
            final_traj_frame = draw_comet_tail(frame_to_draw_bgr.copy(), trajectory_points)
            writer_traj.write(final_traj_frame)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            combined_frame = np.hstack((final_traj_frame, heatmap_color))
            writer_comp.write(combined_frame)

    # --- ✨✨✨ 终极并行流水线 ✨✨✨
    batch_tensors, batch_frames, batch_indices = [], [], []
    with mp.Pool(processes=args.num_workers) as pool:
        pbar = tqdm(total=total_frames - 2, desc="Total Progress")
        
        # "经理" (pool.imap) 现在收发的是大货箱 (chunks)
        for results_chunk in pool.imap(worker_preprocess_chunk, frame_chunk_generator(video_path, total_frames, args.chunk_size)):
            for frame_idx, image_np, frame_to_draw in results_chunk:
                # 主厨在这里把收到的半成品做最后处理 (转Tensor)
                tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().div(255).unsqueeze(0)
                batch_tensors.append(tensor)
                batch_frames.append(frame_to_draw)
                batch_indices.append(frame_idx)
                
                # GPU的批处理逻辑保持不变
                if len(batch_tensors) == args.batch_size:
                    _process_gpu_batch(batch_tensors, batch_frames, batch_indices)
                    batch_tensors.clear(); batch_frames.clear(); batch_indices.clear()

            pbar.update(len(results_chunk))

    # 处理最后一批不满的GPU batch
    if batch_tensors:
        _process_gpu_batch(batch_tensors, batch_frames, batch_indices)
    pbar.close()

    # ... (写入CSV和收尾工作的代码不变) ...
    detection_ratio = (detected_frames_count / total_frames) if total_frames > 0 else 0
    with open(csv_path, 'w', newline='') as f:
        csv_data.sort(key=lambda x: x['frame_number'])
        writer = csv.DictWriter(f, fieldnames=['frame_number', 'detected', 'x', 'y'])
        writer.writeheader()
        writer.writerows(csv_data)
        f.write("\n# --- Summary ---\n")
        f.write(f"total_detected_frame,{detected_frames_count}\n")
        f.write(f"detection_ratio,{detection_ratio:.4f}\n")
    writer_traj.release()
    writer_comp.release()
    print(f"✅ Finished processing. Results saved in: {video_output_dir}")


# --- 5. 总调度室 (新增 chunk_size 参数) ---
def main():
    parser = argparse.ArgumentParser(description="TrackNetV5 Batch Inference Pipeline (V7 - Ultimate Chunked Parallel Edition)")
    parser.add_argument('input_dir', type=str, help='Path to the directory containing input videos.')
    parser.add_argument('weights_path', type=str, help='Path to the model weights (.pth file).')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for inference.')
    parser.add_argument('--batch-size', type=int, default=32, help='GPU batch size for inference.')
    parser.add_argument('--num-workers', type=int, default=max(1, mp.cpu_count() // 2), help='Number of CPU processes for pre-processing.')
    # --- ✨✨✨ 新增的“货箱大小”旋钮 ✨✨✨
    parser.add_argument('--chunk-size', type=int, default=256, help='Number of frames each CPU worker processes at a time.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for detection (0-1).')
    parser.add_argument('--min-circularity', type=float, default=0.7, help='Minimum circularity for a valid detection (0-1).')
    args = parser.parse_args()

    print("🚀 Starting Batch Inference Pipeline (V7 - Ultimate Chunked Parallel Edition)...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    model = build_model(model_cfg)
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu'))
    model.to(device).eval()
    print(f"✅ Model loaded from {args.weights_path} and sent to {device}.")

    input_dir = Path(args.input_dir)
    output_root_dir = input_dir / "utracknet_mvat_wbce"
    output_root_dir.mkdir(exist_ok=True)
    
    video_files = []
    supported_formats = ['*.mp4', '*.mov', '*.MOV', '*.MP4']
    for fmt in supported_formats: video_files.extend(input_dir.glob(fmt))
    if not video_files: print(f"❌ No supported video files found in {input_dir}. Exiting."); return
    video_files = sorted(list(set(video_files)))
    print(f"Found {len(video_files)} videos to process.")
    
    for video_path in video_files:
        process_video(video_path, model, device, args, output_root_dir)

    print(f"\n🎉🎉🎉 All videos processed! Check the results in: {output_root_dir} 🎉🎉🎉")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()