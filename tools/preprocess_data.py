# 文件路径: ./scripts/preprocess_data.py (已修正无球帧的处理逻辑)

import numpy as np
import pandas as pd
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def create_gaussian_kernel(size, variance):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2 + y**2) / float(2 * variance))
    g = g * 255 / g.max()
    return g.astype(np.uint8)

def process_data(input_dir: Path, output_dir: Path, mode: str, config: dict):
    gaussian_kernel = create_gaussian_kernel(config['size'], config['variance'])
    kernel_size = config['size']
    height, width = config['height'], config['width']
    
    label_files = sorted(list(input_dir.glob('**/Label.csv')))
    
    if not label_files:
        print(f"❌ Error: No 'Label.csv' files found in the directory: {input_dir}")
        return

    all_clip_dfs = []
    
    print(f"🚀 Starting data preprocessing for mode: '{mode}'...")
    for label_path in tqdm(label_files, desc="Processing Clips"):
        clip_df = pd.read_csv(label_path)
        clip_root = label_path.parent
        
        gt_clip_output_dir = output_dir / 'gts' / clip_root.relative_to(input_dir)
        gt_clip_output_dir.mkdir(parents=True, exist_ok=True)
        
        gt_paths = []
        # ✨✨✨ 核心改动区域开始 ✨✨✨
        for _, row in clip_df.iterrows():
            gt_path = gt_clip_output_dir / row['file name']
            gt_paths.append(str(gt_path.relative_to(output_dir)))

            # 仅在热力图文件不存在时创建，避免重复工作
            if not gt_path.exists():
                # 1. 首先，创建一个纯黑的画布
                heatmap = np.zeros((height, width), dtype=np.uint8)
                
                # 2. 只有当球可见且坐标存在时，才在画布上画高斯斑点
                if row['visibility'] != 0 and pd.notna(row['x-coordinate']):
                    x, y = int(row['x-coordinate']), int(row['y-coordinate'])
                    
                    x_min, x_max = max(0, x - kernel_size), min(width, x + kernel_size + 1)
                    y_min, y_max = max(0, y - kernel_size), min(height, y + kernel_size + 1)
                    
                    kernel_x_min = max(0, kernel_size - (x - x_min))
                    kernel_x_max = kernel_size + (x_max - x)
                    kernel_y_min = max(0, kernel_size - (y - y_min))
                    kernel_y_max = kernel_size + (y_max - y)

                    if x_max > x_min and y_max > y_min:
                        heatmap[y_min:y_max, x_min:x_max] = gaussian_kernel[kernel_y_min:kernel_y_max, kernel_x_min:kernel_x_max]
                
                # 3. 无论画布上是否有斑点，都将它保存下来
                cv2.imwrite(str(gt_path), heatmap)
        # ✨✨✨ 核心改动区域结束 ✨✨✨

        clip_df['gt_path'] = gt_paths
        base_path_col = clip_root.relative_to(input_dir)
        clip_df['path'] = [str(base_path_col / fname) for fname in clip_df['file name']]
        all_clip_dfs.append(clip_df)
    
    print("✅ All clips processed. Concatenating and creating temporal relationships...")
    master_df = pd.concat(all_clip_dfs, ignore_index=True)
    
    if mode == 'past':
        master_df['path_prev'] = master_df['path'].shift(1)
        master_df['path_preprev'] = master_df['path'].shift(2)
        master_df.dropna(subset=['path_prev', 'path_preprev'], inplace=True)
        final_columns = ['path', 'path_prev', 'path_preprev', 'gt_path', 'x-coordinate', 'y-coordinate', 'status', 'visibility']
    elif mode == 'context':
        master_df['path_prev'] = master_df['path'].shift(1)
        master_df['path_next'] = master_df['path'].shift(-1)
        master_df.dropna(subset=['path_prev', 'path_next'], inplace=True)
        final_columns = ['path', 'path_prev', 'path_next', 'gt_path', 'x-coordinate', 'y-coordinate', 'status', 'visibility']
    
    final_df = master_df[final_columns]
    
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    num_train = int(len(final_df) * config['train_rate'])
    
    df_train = final_df.iloc[:num_train]
    df_val = final_df.iloc[num_train:]
    
    train_csv_path = output_dir / f"labels_{mode}_train.csv"
    val_csv_path = output_dir / f"labels_{mode}_val.csv"
    
    df_train.to_csv(train_csv_path, index=False)
    df_val.to_csv(val_csv_path, index=False)
    
    print(f"🎉 Preprocessing for mode '{mode}' complete!")
    print(f"Train samples: {len(df_train)}, saved to {train_csv_path}")
    print(f"Validation samples: {len(df_val)}, saved to {val_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TrackNet Dataset Preprocessing Script")
    parser.add_argument('--input_dir','-in', type=str, required=True, help='Path to the raw data directory.')
    parser.add_argument('--output_dir','-out', type=str, required=True, help='Path to save the processed data and labels.')
    parser.add_argument('--mode','-m', type=str, required=True, choices=['past', 'context'], help="Processing mode.")
    parser.add_argument('--height', type=int, default=720, help='Target image height.')
    parser.add_argument('--width', type=int, default=1280, help='Target image width.')
    parser.add_argument('--size', type=int, default=20, help='Radius of the Gaussian kernel.')
    parser.add_argument('--variance', type=float, default=10, help='Variance of the Gaussian kernel.')
    parser.add_argument('--train_rate', type=float, default=0.7, help='Proportion of the dataset to use for training.')
    
    args = parser.parse_args()

    config = {
        'height': args.height, 'width': args.width,
        'size': args.size, 'variance': args.variance,
        'train_rate': args.train_rate
    }
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    process_data(input_path, output_path, args.mode, config)