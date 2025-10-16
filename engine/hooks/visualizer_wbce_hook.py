import cv2
import torch
import numpy as np
import math
from pathlib import Path

from .base_hook import BaseHook
from ..builder import HOOKS


try:
    from metrics_factory.metrics.utracknetv1_metric import _heatmap_to_coords
except ImportError:
    print("Warning: _heatmap_to_coords not found. Visualization will not show predicted coordinates.")
    def _heatmap_to_coords(heatmap, threshold):
        return None, None


@HOOKS.register_module
class ValidationVisualizerHookWBCE(BaseHook):
    """
    一个在验证阶段，用于将模型预测结果可视化的钩子。
    它会将 输入-预测-真值 对比图保存到实验目录下。
    """

    def __init__(self, num_samples_to_save=5, heatmap_threshold=127, original_size=(360, 640)):
        self.num_samples_to_save = num_samples_to_save
        self.heatmap_threshold = heatmap_threshold
        self.original_h, self.original_w = original_size
        self.vis_count = 0

    def before_val_epoch(self, runner):
        """在每次验证epoch开始前，重置计数器。"""
        self.vis_count = 0
        # 确保可视化结果的保存目录存在
        self.vis_dir = runner.work_dir / f'val_epoch_{runner.epoch + 1}'
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def after_val_iter(self, runner):
        """在每次验证迭代后，检查是否需要进行可视化。"""
        if self.vis_count >= self.num_samples_to_save:
            return

        # 从 runner 中获取数据
        batch = runner.outputs['val_batch']
        logits = runner.outputs['val_logits']

        # --- 以下是您原有的数据处理逻辑，保持不变 ---
        input_tensor = batch['image'].cpu()   # (B, C, H, W)
        target_tensor = batch['target'].cpu()   # (B, H, W)
        pred_tensor = logits.squeeze(dim=1).cpu() * 255 # (B, H, W)

        # 遍历这个批次中的每一张图
        for i in range(input_tensor.size(0)):
            if self.vis_count >= self.num_samples_to_save:
                break

            # --- 构建可视化图像 (这部分逻辑也保持不变) ---
            input_frame_rgb = (input_tensor[i, 5:8, :, :].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            input_frame_bgr = cv2.cvtColor(input_frame_rgb, cv2.COLOR_RGB2BGR)

            pred_heatmap_np = pred_tensor[i].numpy().astype(np.uint8)
            target_heatmap_np = target_tensor[i].numpy().astype(np.uint8)

            pred_heatmap_color = cv2.applyColorMap(pred_heatmap_np, cv2.COLORMAP_JET)
            target_heatmap_color = cv2.applyColorMap(target_heatmap_np, cv2.COLORMAP_JET)

            h, w, _ = input_frame_bgr.shape

            # ✨ --- 新增代码开始：获取、缩放并绘制坐标 --- ✨
            
            # 假设 batch['coords'] 存在，并且 self.original_w, self.original_h 已在 __init__ 中定义
            coords_gt_batch = batch['coords']
            x_gt_raw, y_gt_raw = coords_gt_batch[0][i].item(), coords_gt_batch[1][i].item()

            # 从预测热力图中提取坐标
            x_pred, y_pred = _heatmap_to_coords(pred_heatmap_np, threshold=self.heatmap_threshold)

            # 绘制绿色的真实坐标 (需要进行缩放)
            if not math.isnan(x_gt_raw) and not math.isnan(y_gt_raw):
                x_gt_scaled = int(x_gt_raw * (w / self.original_w))
                y_gt_scaled = int(y_gt_raw * (h / self.original_h))
                cv2.drawMarker(input_frame_bgr, (x_gt_scaled, y_gt_scaled), color=(0, 255, 0), 
                               markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

            # 绘制红色的预测坐标
            if x_pred is not None and y_pred is not None:
                cv2.drawMarker(input_frame_bgr, position=(int(x_pred), int(y_pred)), color=(0, 0, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

            # ✨ --- 新增代码结束 --- ✨

            # --- 拼接画布 (这部分逻辑也保持不变) ---
            canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
            canvas[:, 0:w] = input_frame_bgr
            canvas[:, w:2 * w] = pred_heatmap_color
            canvas[:, 2 * w:3 * w] = target_heatmap_color

            # ✨ 修改文字标签以说明十字的颜色 ✨
            cv2.putText(canvas, 'Input (Red:Pred, Green:GT)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, 'Prediction', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(canvas, 'Ground Truth', (2 * w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 保存图像
            save_path = self.vis_dir / f'sample_{self.vis_count}.jpg'
            cv2.imwrite(str(save_path), canvas)
            self.vis_count += 1