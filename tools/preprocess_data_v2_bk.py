# 文件路径: ./scripts/preprocess_data.py (已修正无球帧的处理逻辑)

import numpy as np
import pandas as pd
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm



def get_heatmap(h, w, cx, cy, r):

    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    # print(f'{x.shape}, {y.shape}')

    heatmap = ((y - (cy + 1)) ** 2) + ((x - (cx + 1)) ** 2)
    heatmap[heatmap <= r ** 2] = 1
    heatmap[heatmap > r ** 2] = 0
    return heatmap.astype(np.uint8) * 255


def process_data(input_dir: Path, output_dir: Path, mode: str, config: dict):
    gt_height, gt_width = config['gt_height'], config['gt_width']
    height, width = config['height'], config['width']

    ratio = gt_height / height
    radius = config['radius']

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
                heatmap = np.zeros((gt_height, gt_width), dtype=np.uint8)

                # 2. 只有当球可见且坐标存在时，才在画布上画高斯斑点
                if row['visibility'] != 0 and pd.notna(row['x-coordinate']):
                    x, y = int(row['x-coordinate']), int(row['y-coordinate'])
                    heatmap = get_heatmap(gt_height, gt_width, x * ratio, y * ratio, radius)


                # 3. 无论画布上是否有斑点，都将它保存下来
                cv2.imwrite(str(gt_path), heatmap)
        # ✨✨✨ 核心改动区域结束 ✨✨✨

        clip_df['gt_path'] = gt_paths
        base_path_col = clip_root.relative_to(input_dir)
        clip_df['path'] = [str(base_path_col / fname) for fname in clip_df['file name']]

        # ✨✨✨ 新增：为每帧添加前后帧的信息 ✨✨✨
        if mode == 'past':
            # 添加前一帧和后一帧的路径和标签信息
            clip_df['path_prev'] = clip_df['path'].shift(1)
            clip_df['path_next'] = clip_df['path'].shift(-1)

            # 添加前一帧和后一帧的gt路径
            clip_df['gt_path_prev'] = clip_df['gt_path'].shift(1)
            clip_df['gt_path_next'] = clip_df['gt_path'].shift(-1)

            # 添加前一帧和后一帧的坐标信息
            clip_df['x_prev'] = clip_df['x-coordinate'].shift(1)
            clip_df['y_prev'] = clip_df['y-coordinate'].shift(1)
            clip_df['x_next'] = clip_df['x-coordinate'].shift(-1)
            clip_df['y_next'] = clip_df['y-coordinate'].shift(-1)

            # 添加前一帧和后一帧的visibility和status
            clip_df['visibility_prev'] = clip_df['visibility'].shift(1)
            clip_df['status_prev'] = clip_df['status'].shift(1)
            clip_df['visibility_next'] = clip_df['visibility'].shift(-1)
            clip_df['status_next'] = clip_df['status'].shift(-1)

        elif mode == 'context':
            # 添加前一帧和后一帧的路径和标签信息
            clip_df['path_prev'] = clip_df['path'].shift(1)
            clip_df['path_next'] = clip_df['path'].shift(-1)

            # 添加前一帧和后一帧的gt路径
            clip_df['gt_path_prev'] = clip_df['gt_path'].shift(1)
            clip_df['gt_path_next'] = clip_df['gt_path'].shift(-1)

            # 添加前一帧和后一帧的坐标信息
            clip_df['x_prev'] = clip_df['x-coordinate'].shift(1)
            clip_df['y_prev'] = clip_df['y-coordinate'].shift(1)
            clip_df['x_next'] = clip_df['x-coordinate'].shift(-1)
            clip_df['y_next'] = clip_df['y-coordinate'].shift(-1)

            # 添加前一帧和后一帧的visibility和status
            clip_df['visibility_prev'] = clip_df['visibility'].shift(1)
            clip_df['status_prev'] = clip_df['status'].shift(1)
            clip_df['visibility_next'] = clip_df['visibility'].shift(-1)
            clip_df['status_next'] = clip_df['status'].shift(-1)

        # 删除首尾帧（因为它们没有完整的前后帧）
        clip_df = clip_df.iloc[1:-1]

        all_clip_dfs.append(clip_df)

    print("✅ All clips processed. Concatenating and creating temporal relationships...")
    master_df = pd.concat(all_clip_dfs, ignore_index=True)

    # ✨✨✨ 修改最终的列选择 ✨✨✨
    if mode == 'past':
        final_columns = [
            'path_prev', 'path', 'path_next',  # 三张图片路径：前一帧、当前帧、后一帧
            'gt_path_prev', 'gt_path', 'gt_path_next',  # 三张对应的gt图路径
            'x_prev', 'y_prev', 'x-coordinate', 'y-coordinate', 'x_next', 'y_next',  # 三个x,y坐标
            'visibility_prev', 'visibility', 'visibility_next',  # 三个visibility
            'status_prev', 'status', 'status_next'  # 三个status
        ]
    elif mode == 'context':
        final_columns = [
            'path_prev', 'path', 'path_next',  # 三张图片路径：前一帧、当前帧、后一帧
            'gt_path_prev', 'gt_path', 'gt_path_next',  # 三张对应的gt图路径
            'x_prev', 'y_prev', 'x-coordinate', 'y-coordinate', 'x_next', 'y_next',  # 三个x,y坐标
            'visibility_prev', 'visibility', 'visibility_next',  # 三个visibility
            'status_prev', 'status', 'status_next'  # 三个status
        ]

    final_df = master_df[final_columns]

    # 重命名列以保持一致性
    column_rename = {
        'x-coordinate': 'x_current',
        'y-coordinate': 'y_current',
        'visibility': 'visibility_current',
        'status': 'status_current'
    }
    final_df = final_df.rename(columns=column_rename)

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

    # 打印第一行数据作为示例
    print("\n📊 Example of first row in final dataset:")
    print(f"Image paths: {df_train.iloc[0]['path_prev']}, {df_train.iloc[0]['path']}, {df_train.iloc[0]['path_next']}")
    print(
        f"GT paths: {df_train.iloc[0]['gt_path_prev']}, {df_train.iloc[0]['gt_path']}, {df_train.iloc[0]['gt_path_next']}")
    print(
        f"Coordinates: ({df_train.iloc[0]['x_prev']}, {df_train.iloc[0]['y_prev']}), ({df_train.iloc[0]['x_current']}, {df_train.iloc[0]['y_current']}), ({df_train.iloc[0]['x_next']}, {df_train.iloc[0]['y_next']})")
    print(
        f"Visibility: {df_train.iloc[0]['visibility_prev']}, {df_train.iloc[0]['visibility_current']}, {df_train.iloc[0]['visibility_next']}")
    print(
        f"Status: {df_train.iloc[0]['status_prev']}, {df_train.iloc[0]['status_current']}, {df_train.iloc[0]['status_next']}")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="TrackNet Dataset Preprocessing Script")
    # parser.add_argument('--input_dir', '-in', type=str, required=True, help='Path to the raw data directory.')
    # parser.add_argument('--output_dir', '-out', type=str, required=True,
    #                     help='Path to save the processed data and labels.')
    # parser.add_argument('--mode', '-m', type=str, required=True, choices=['past', 'context'], help="Processing mode.")
    # parser.add_argument('--height', type=int, default=720, help='Target image height.')
    # parser.add_argument('--width', type=int, default=1280, help='Target image width.')
    # parser.add_argument('--gt_height', '-gt_h', type=int, default=288, help='heatmap height.')
    # parser.add_argument('--gt_width', '-gt_w', type=float, default=512, help='heatmap width.')
    # parser.add_argument('--radius', type=float, default=2.5, help='radius of heatmap position.')
    # parser.add_argument('--train_rate', type=float, default=0.7, help='Proportion of the dataset to use for training.')
    #
    # args = parser.parse_args()

    config = {
        'height': 720,
        'width': 1280,
        'train_rate': 0.7,
        'gt_height': 288,
        'gt_width': 512,
        'radius': 2.5
    }

    input_path = Path('E:\\tracknet\TracknetV5\data\\v2')
    output_path = Path('E:\\tracknet\TracknetV5\data\\v2')

    process_data(input_path, output_path, 'context', config)