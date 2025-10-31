import os.path

import torch
import cv2
import numpy as np
import math
from pathlib import Path
import importlib.util
from torch.utils.data import DataLoader
from tqdm import tqdm
import models_factory
import datasets_factory
from metrics_factory.metrics.utracknetv1_metric import _heatmap_to_coords

def load_config_from_path(config_path: str):
    """从 .py 文件路径中加载配置模块。"""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # 将 .py 文件作为模块加载
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    return cfg_module

vis_count = 0

def visibility(data:dict, heatmap_threshold=127, original_size=(720, 1280), output_dir='./output'):
    '''
    可视化测试集
    '''
    global vis_count
    original_h, original_w = original_size

    logits = data['logits']
    draft = data['draft'].cpu()
    input_tensor = data['image'].cpu()
    target_tensor = data['target'].cpu()
    coords_gt_batch = data['coords']
    # print(logits.shape)
    pred_tensor = logits.cpu() * 255
    draft_tensor = draft.cpu() * 255

    for i in range(input_tensor.size(0)):

        # --- 准备三帧原图 ---
        # 输入形状为 [b, 9, h, w]，因为三帧RGB图像 (3帧 * 3通道)
        input_frames = []
        for frame_idx in range(3):  # 处理三帧
            # 获取当前帧的RGB通道 (3个通道)
            frame_rgb = input_tensor[i, frame_idx * 3:(frame_idx + 1) * 3, :, :].permute(1, 2, 0).numpy()
            frame_rgb = (frame_rgb * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            input_frames.append(frame_bgr)
        h, w, _ = input_frames[0].shape  # 获取单帧尺寸

        # --- 准备GT热力图 ---
        # target_tensor形状为 [b, 3, h, w]，对应三帧的GT热力图
        gt_heatmaps = []
        for frame_idx in range(3):
            gt_heatmap = target_tensor[i, frame_idx].numpy().astype(np.uint8)
            gt_heatmap_color = cv2.applyColorMap(gt_heatmap, cv2.COLORMAP_JET)
            gt_heatmaps.append(gt_heatmap_color)

        # --- 准备预测热力图（三帧）---
        pred_heatmaps = []
        pred_coords = []  # 存储每帧的预测坐标
        for frame_idx in range(3):
            pred_heatmap_np = pred_tensor[i, frame_idx].numpy().astype(np.uint8)
            pred_heatmap_color = cv2.applyColorMap(pred_heatmap_np, cv2.COLORMAP_JET)
            pred_heatmaps.append(pred_heatmap_color)

            # 获取每帧的预测坐标
            x_pred, y_pred = _heatmap_to_coords(pred_heatmap_np, threshold=heatmap_threshold)
            pred_coords.append((x_pred, y_pred))

        # --- 准备修复前热力图（三帧）---
        draft_heatmaps = []
        for frame_idx in range(3):
            draft_heatmap_np = draft_tensor[i, frame_idx].numpy().astype(np.uint8)
            draft_heatmap_color = cv2.applyColorMap(draft_heatmap_np, cv2.COLORMAP_JET)
            draft_heatmaps.append(draft_heatmap_color)

        # --- 准备差异热力图（三帧）---
        diff_heatmaps = []
        for frame_idx in range(3):
            diff_heatmap_np = (pred_tensor[i, frame_idx] - draft_tensor[i, frame_idx]).numpy()
            # 此时，diff=0 (零差异) 将映射到 127.5 (颜色条中心)
            normalized_diff = (diff_heatmap_np + 255.0) / (2 * 255.0) * 255.0

            # 截断并转换为 uint8 (这是 cv2.applyColorMap 要求的输入格式)
            normalized_diff_uint8 = np.clip(normalized_diff, 0, 255).astype(np.uint8)
            diff_heatmap_color = cv2.applyColorMap(normalized_diff_uint8, cv2.COLORMAP_JET)
            diff_heatmaps.append(diff_heatmap_color)

        # --- 准备准确点 ---
        x_gt = []
        y_gt = []
        for frame_idx in range(3):
            x_gt_raw = coords_gt_batch[frame_idx][0][i].item()
            y_gt_raw = coords_gt_batch[frame_idx][1][i].item()
            x_gt.append(x_gt_raw)
            y_gt.append(y_gt_raw)

        # 在所有帧原图上绘制标记
        frames_with_marks = []
        for frame_idx in range(3):
            input_img = input_frames[frame_idx]

            # 绘制绿色的真实标记
            if not math.isnan(x_gt[frame_idx]) and not math.isnan(y_gt[frame_idx]):
                x_gt_scaled = int(x_gt[frame_idx] * (w / original_w))
                y_gt_scaled = int(y_gt[frame_idx] * (h / original_h))
                cv2.drawMarker(input_img, (x_gt_scaled, y_gt_scaled),
                               color=(0, 255, 0), markerType=cv2.MARKER_CROSS,
                               markerSize=15, thickness=2)

            # 绘制红色的预测标记
            x_pred, y_pred = pred_coords[frame_idx]
            if x_pred is not None:
                cv2.drawMarker(input_img, (int(x_pred), int(y_pred)),
                               color=(0, 0, 255), markerType=cv2.MARKER_CROSS,
                               markerSize=15, thickness=2)


        # --- 拼接画布 ---
        canvas = np.zeros((h * 3, w * 5, 3), dtype=np.uint8)
        for frame_idx in range(3):
            canvas[frame_idx * h:(frame_idx+1) * h, 0:w] = input_frames[frame_idx]
            canvas[frame_idx * h:(frame_idx+1) * h, w:2 * w] = pred_heatmaps[frame_idx]
            canvas[frame_idx * h:(frame_idx+1) * h, 2 * w:3 * w] = draft_heatmaps[frame_idx]
            canvas[frame_idx * h:(frame_idx + 1) * h, 3 * w:4 * w] = diff_heatmaps[frame_idx]
            canvas[frame_idx * h:(frame_idx + 1) * h, 4 * w:5 * w] = gt_heatmaps[frame_idx]

        cv2.putText(canvas, 'Input (Red:Pred, Green:GT)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                    2)
        cv2.putText(canvas, 'Prediction', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, 'draft', (2 * w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, 'difference', (3 * w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, 'GT True', (4 * w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 保存图像
        save_path = os.path.join(output_dir,f'sample_{vis_count}.jpg')
        cv2.imwrite(str(save_path), canvas)
        vis_count += 1

if __name__ == "__main__":
    cfg = load_config_from_path('./configs/experiments/tracknetv5_test.py')
    print("✅ Configuration loaded successfully.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 构建模型
    model = models_factory.build_model(cfg.model)
    print("✅ Model built successfully.")

    # 加载权重
    model.load_state_dict(torch.load(cfg.model_pth_path, map_location='cpu'))
    # 创建输出文件夹
    output_dir = cfg.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出文件夹{output_dir}")

    # 获取数据集
    val_dataset = datasets_factory.build_dataset(cfg.data['val'])
    print("✅ Datasets built successfully.")

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.data['samples_per_gpu'],
        num_workers=cfg.data['workers_per_gpu'],
        shuffle=False,
        pin_memory=True
    )

    model.eval().to(device)
    data = {}
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input = batch['image'].to(device)
            logits, draft = model(input)
            data['image'] = batch['image']
            data['target'] = batch['target']
            data['logits'] = logits
            data['draft'] = draft
            data['coords'] = batch['coords']
            visibility(data,output_dir=output_dir, original_size=cfg.original_size)