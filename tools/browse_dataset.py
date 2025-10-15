import argparse
from pathlib import Path
import cv2
import numpy as np
import sys
import re

# 导入我们的工厂和配置
# 导入顶层包，__init__.py会自动触发所有模块的注册
import datasets_factory
import models_factory 

# 从 configs 目录导入我们写好的“生产订单”
# 请确保这个路径与您的配置文件路径一致
from configs.experiments.utracknetv1_mvat_tennis_b2e500 import data as data_cfg

def main():
    # 智能地从配置文件决定运行模式
    pipeline_types = [p['type'] for p in data_cfg['train']['pipeline']]
    is_attention_mode = 'GenerateMotionAttention' in pipeline_types
    
    # 1. 根据配置文件构建数据集
    print("Building dataset from config...")
    dataset = datasets_factory.build_dataset(data_cfg['train'])
    
    print("\n" + "="*50)
    print(f"🚀 Starting Interactive Dataset Browser...")
    print(f"   Mode: {'Motion Attention' if is_attention_mode else 'Default'}")
    print("   Press ANY KEY to advance to the next frame.")
    print("   Press 'q' or ESC to quit.")
    print("="*50)

    for i, sample in enumerate(dataset):
        image_tensor = sample['image'] # (C, H, W) Tensor, 值范围 [0.0, 1.0]
        target_tensor = sample['target'] # (H, W) Tensor
        original_info = sample['original_info']
        
        # --- 将拼接后的Tensor智能地拆分回可视化组件 ---
        
        input_frames, attention_maps = [], []
        frame_labels, att_labels = [], []
        h, w = image_tensor.shape[1], image_tensor.shape[2]

        if is_attention_mode:
            # 顺序: img_prev(3), att1(2), img_curr(3), att2(2), img_next(3) -> 13通道
            input_frames.extend([
                image_tensor[0:3],   # prev
                image_tensor[5:8],   # curr
                image_tensor[10:13]  # next
            ])
            attention_maps.extend([
                image_tensor[3:5],   # att_prev_to_curr
                image_tensor[8:10]   # att_curr_to_next
            ])
            paths = [original_info[col] for col in ['path_prev', 'path', 'path_next']]
            frame_labels = [Path(p).name for p in paths]
            att_labels = [f"Att: {frame_labels[0]}->{frame_labels[1]}", f"Att: {frame_labels[1]}->{frame_labels[2]}"]
        else: # 默认模式
            # 顺序: img_prev(3), img_curr(3), img_next(3) -> 9通道
            input_frames.extend([
                image_tensor[0:3],
                image_tensor[3:6],
                image_tensor[6:9]
            ])
            frame_labels = [Path(original_info[col]).name for col in ['path_prev', 'path', 'path_next']]

        # --- 可视化逻辑 ---
        
        # 准备所有要显示的BGR格式图像
        frames_bgr = []
        for frame_tensor in input_frames:
            frame_np_rgb = (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frames_bgr.append(cv2.cvtColor(frame_np_rgb, cv2.COLOR_RGB2BGR))

        att_maps_color = []
        if is_attention_mode:
            for att_tensor in attention_maps:
                att_np = att_tensor.permute(1, 2, 0).numpy()
                att_map_color = np.dstack([att_np[:,:,1], np.zeros_like(att_np[:,:,0]), att_np[:,:,0]]) * 255
                att_maps_color.append(att_map_color.astype(np.uint8))
        
        gt_color = cv2.applyColorMap(target_tensor.numpy().astype(np.uint8), cv2.COLORMAP_JET)

        # 创建大画布
        vis_h, vis_w = h, w
        header_h = 40
        canvas = np.zeros((vis_h * 3 + header_h, vis_w * 3, 3), dtype=np.uint8)

        # 绘制左侧输入帧
        for j, frame in enumerate(frames_bgr):
            canvas[j*vis_h+header_h:(j+1)*vis_h+header_h, 0:vis_w] = frame
            cv2.putText(canvas, f"Input: {frame_labels[j]}", (10, j*vis_h+header_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # 绘制中间的注意力图
        if is_attention_mode:
            for j, att_map in enumerate(att_maps_color):
                canvas[j*vis_h+header_h:(j+1)*vis_h+header_h, vis_w:2*vis_w] = att_map
                cv2.putText(canvas, att_labels[j], (vis_w+10, j*vis_h+header_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # 绘制右侧GT (居中放置不拉伸)
        gt_display = cv2.resize(gt_color, (vis_w, vis_h))
        offset_y = header_h + vis_h
        canvas[offset_y : offset_y + vis_h, 2*vis_w : 3*vis_w] = gt_display
        cv2.putText(canvas, "Ground Truth", (2*vis_w+10, header_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # 绘制标题
        title = f"Sample #{i+1}/{len(dataset)} | Visibility: {original_info.get('visibility', 'N/A')}"
        cv2.putText(canvas, title, (10, header_h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        # 在显示前，按比例缩小最终的画布
        scale_factor = 0.5 # 您可以根据屏幕大小调整这个比例
        small_canvas = cv2.resize(canvas, (0, 0), fx=scale_factor, fy=scale_factor)

        cv2.imshow('TrackNet Dataset Browser (Deluxe Edition)', small_canvas)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
    print("\nBrowser closed.")

if __name__ == '__main__':
    main()