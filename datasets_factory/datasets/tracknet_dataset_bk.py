# 文件路径: ./datasets/tracknet_dataset.py (真正修正版)

import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from torchvision import transforms
import re
import argparse
import sys

class TrackNetDataset(Dataset):
    """
    一个经过专业重构的、支持多种模式的数据集类。
    """
    def __init__(self, data_dir, csv_path, 
                 use_motion_attention: bool = False, 
                 attention_threshold: int = 40,
                 input_height=360, input_width=640):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.csv_path = Path(csv_path)
        self.use_motion_attention = use_motion_attention
        self.attention_threshold = attention_threshold
        
        self.data = pd.read_csv(self.csv_path)
        print(f"Loaded {self.csv_path.name}, samples = {len(self.data)}")
        
        self.height = input_height
        self.width = input_width
        
        if self.use_motion_attention:
            print("🔥 Motion Attention mode is ENABLED.")
            self.input_path_cols = ['path_prev', 'path', 'path_next']
            if not all(col in self.data.columns for col in self.input_path_cols):
                raise ValueError(f"Motion Attention mode requires columns: {self.input_path_cols}")
        else:
            print("Default mode is ENABLED.")
            all_columns = self.data.columns
            self.input_path_cols = sorted([col for col in all_columns if 'path' in col and col != 'gt_path'])
        
        print(f"Detected input path columns for this mode: {self.input_path_cols}")
        if len(self.input_path_cols) != 3:
            raise ValueError(f"Expected 3 input path columns, but found {len(self.input_path_cols)}: {self.input_path_cols}")
            
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _get_frame_number(path_str):
        match = re.search(r'(\d+)\.jpg', str(path_str))
        return int(match.group(1)) if match else -1

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        if not self.use_motion_attention:
            images_to_stack = []
            for col_name in self.input_path_cols:
                img_path = self.data_dir / row[col_name]
                img = cv2.imread(str(img_path))
                img = cv2.resize(img, (self.width, self.height))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images_to_stack.append(img)
            
            stacked_imgs_np = np.concatenate(images_to_stack, axis=2)
            inputs = self.input_transform(stacked_imgs_np)
        else:
            # 加载 attention 模式所需的三帧
            path_prev = self.data_dir / row['path_prev']
            path_curr = self.data_dir / row['path']
            path_next = self.data_dir / row['path_next']

            img_prev = cv2.resize(cv2.imread(str(path_prev)), (self.width, self.height))
            img_curr = cv2.resize(cv2.imread(str(path_curr)), (self.width, self.height))
            img_next = cv2.resize(cv2.imread(str(path_next)), (self.width, self.height))
            
            def get_attention_map(frame1, frame2, threshold):
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.int16)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.int16)
                diff = gray2 - gray1
                
                # ✨✨✨ 最终、决定性的修正！ ✨✨✨
                brighten_map = ((diff > threshold) * 255).astype(np.uint8)
                darken_map = ((diff < -threshold) * 255).astype(np.uint8)
                
                return np.stack([brighten_map, darken_map], axis=-1)

            att_prev_to_curr = get_attention_map(img_prev, img_curr, self.attention_threshold)
            att_curr_to_next = get_attention_map(img_curr, img_next, self.attention_threshold)
            
            img_prev_rgb = cv2.cvtColor(img_prev, cv2.COLOR_BGR2RGB)
            img_curr_rgb = cv2.cvtColor(img_curr, cv2.COLOR_BGR2RGB)
            img_next_rgb = cv2.cvtColor(img_next, cv2.COLOR_BGR2RGB)
            
            final_input_np = np.concatenate([
                img_prev_rgb,
                att_prev_to_curr,
                img_curr_rgb,
                att_curr_to_next,
                img_next_rgb
            ], axis=2)
            
            inputs = self.input_transform(final_input_np)

        gt_path = self.data_dir / row['gt_path']
        target_np = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        target_np = cv2.resize(target_np, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        target = torch.from_numpy(target_np.astype(np.int64))

        return {
            'image': inputs,
            'target': target,
            'coords': (row['x-coordinate'], row['y-coordinate']),
            'visibility': row['visibility']
        }

# ==============================================================================
# ======================== 交互式浏览器与测试代码 =================================
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive TrackNet Dataset Browser")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the unified data directory (e.g., './raw_data').")
    parser.add_argument('--mode', type=str, default='past', choices=['past', 'context'], help="Which dataset mode to browse.")
    parser.add_argument('--attention', action='store_true', help="Enable the motion attention 'enhancer' mode.")
    args = parser.parse_args()

    if args.attention:
        csv_name = f"labels_context_train.csv"
    else:
        csv_name = f"labels_{args.mode}_train.csv"
    csv_path = Path(args.data_dir) / csv_name
    
    if not csv_path.exists():
        print(f"❌ Error: Could not find '{csv_path.name}' in '{args.data_dir}'.")
        print("Please run the preprocessing script first for this mode.")
        sys.exit(1)

    VIS_HEIGHT, VIS_WIDTH = 270, 480
    
    dataset = TrackNetDataset(
        data_dir=args.data_dir,
        csv_path=csv_path,
        use_motion_attention=args.attention,
        input_height=VIS_HEIGHT, input_width=VIS_WIDTH
    )

    print("\n" + "="*50)
    print("🚀 Starting Interactive Dataset Browser...")
    print("   Press ANY KEY to advance to the next frame.")
    print("   Press 'q' or ESC to quit.")
    print("="*50)

    for i in range(len(dataset)):
        sample = dataset[i]
        image_tensor = sample['image']
        
        input_frames, frame_labels = [], []
        if args.attention:
            input_frames.extend([image_tensor[0:3], image_tensor[5:8], image_tensor[10:13]])
            row = dataset.data.iloc[i]
            paths = [row[col] for col in dataset.input_path_cols]
            sorted_paths = sorted(paths, key=lambda p: int(re.search(r'(\d+)\.jpg', str(p)).group(1)))
            frame_labels = [Path(p).name for p in sorted_paths]
        else:
            input_frames.extend([image_tensor[0:3], image_tensor[3:6], image_tensor[6:9]])
            frame_labels = dataset.input_path_cols

        attention_maps, att_labels = [], []
        if args.attention:
            att1 = image_tensor[3:5].permute(1,2,0).numpy()
            att2 = image_tensor[8:10].permute(1,2,0).numpy()
            att_map1_color = np.dstack([att1[:,:,1], np.zeros_like(att1[:,:,0]), att1[:,:,0]]) * 255
            att_map2_color = np.dstack([att2[:,:,1], np.zeros_like(att2[:,:,0]), att2[:,:,0]]) * 255
            attention_maps.extend([att_map1_color, att_map2_color])
            att_labels = [f"Att: {frame_labels[0]}->{frame_labels[1]}", f"Att: {frame_labels[1]}->{frame_labels[2]}"]

        gt_np = sample['target'].numpy()
        gt_color = cv2.applyColorMap(gt_np.astype(np.uint8), cv2.COLORMAP_JET)

        header_h = 40
        canvas = np.zeros((VIS_HEIGHT * 3 + header_h, VIS_WIDTH * 3, 3), dtype=np.uint8)
        
        for j, frame_tensor in enumerate(input_frames):
            frame_np = (frame_tensor.permute(1,2,0).numpy() * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            canvas[j*VIS_HEIGHT+header_h:(j+1)*VIS_HEIGHT+header_h, 0:VIS_WIDTH] = frame_bgr
            cv2.putText(canvas, f"Input: {frame_labels[j]}", (10, j*VIS_HEIGHT+header_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        if args.attention:
            for j, att_map in enumerate(attention_maps):
                canvas[j*VIS_HEIGHT+header_h:(j+1)*VIS_HEIGHT+header_h, VIS_WIDTH:2*VIS_WIDTH] = att_map
                cv2.putText(canvas, att_labels[j], (VIS_WIDTH+10, j*VIS_HEIGHT+header_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        right_panel = np.zeros((VIS_HEIGHT * 3, VIS_WIDTH, 3), dtype=np.uint8)
        offset_y = (VIS_HEIGHT * 3 - VIS_HEIGHT) // 2
        right_panel[offset_y:offset_y+VIS_HEIGHT, :] = cv2.resize(gt_color, (VIS_WIDTH, VIS_HEIGHT))
        canvas[header_h:, 2*VIS_WIDTH:3*VIS_WIDTH] = right_panel
        cv2.putText(canvas, "Ground Truth", (2*VIS_WIDTH+10, header_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        title = f"Sample #{i+1}/{len(dataset)} | Mode: {'Motion Attention' if args.attention else args.mode.upper()}"
        cv2.putText(canvas, title, (10, header_h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow('TrackNet Dataset Browser', canvas)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break
    
    cv2.destroyAllWindows()
    print("\nBrowser closed.")