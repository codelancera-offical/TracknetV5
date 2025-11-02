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
class ValidationVisualizerHookV1(BaseHook):
    """
    一个在验证阶段，用于将模型预测结果（包括最终坐标）可视化的钩子。
    """
    def __init__(self, num_samples_to_save=5, heatmap_threshold=127, original_size=(720, 1280)):
        self.num_samples_to_save = num_samples_to_save
        self.heatmap_threshold = heatmap_threshold
        self.original_h, self.original_w = original_size # 存储原始视频尺寸 (高, 宽)
        self.vis_count = 0

    def before_val_epoch(self, runner):
        """在每次验证epoch开始前，重置计数器。"""
        self.vis_count = 0
        self.vis_dir = runner.work_dir / f'val_epoch_{runner.epoch + 1}'
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def after_val_iter(self, runner):
        """在每次验证迭代后，检查是否需要进行可视化。"""
        if self.vis_count >= self.num_samples_to_save:
            return

        # 从 runner 中获取数据
        batch = runner.outputs['val_batch']
        logits = runner.outputs['val_logits']
        
        input_tensor = batch['image'].cpu()
        target_tensor = batch['target'].cpu()
        coords_gt_batch = batch['coords']
        print(logits.shape)

        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_tensor = torch.argmax(probs, dim=1).cpu() # 现在就是0-255了
    
        
        for i in range(input_tensor.size(0)):
            if self.vis_count >= self.num_samples_to_save:
                break
            
            # --- 准备基础可视化图像 ---
            input_frame_rgb = (input_tensor[i, 0:3, :, :].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            input_frame_bgr = cv2.cvtColor(input_frame_rgb, cv2.COLOR_RGB2BGR)
            h, w, _ = input_frame_bgr.shape # 获取【当前显示】的图像尺寸 (例如 360, 640)

            pred_heatmap_np = pred_tensor[i].numpy().astype(np.uint8)
            print(f"DEBUG: Shape of pred_heatmap_np is {pred_heatmap_np.shape}") 
            pred_heatmap_color = cv2.applyColorMap(pred_heatmap_np, cv2.COLORMAP_JET)
            target_heatmap_color = cv2.applyColorMap(target_tensor[i].numpy().astype(np.uint8), cv2.COLORMAP_JET)
            
            # --- 获取并缩放坐标 ---
            x_pred, y_pred = _heatmap_to_coords(pred_heatmap_np, threshold=self.heatmap_threshold)
            
            # 从batch中获取原始的、高分辨率下的GT坐标
            x_gt_raw, y_gt_raw = coords_gt_batch[0][i].item(), coords_gt_batch[1][i].item()

            # ✨✨✨ 关键修正：在绘制前，进行坐标缩放 ✨✨✨
            # 只有当原始坐标是有效数字时，才进行缩放和绘制
            if not math.isnan(x_gt_raw) and not math.isnan(y_gt_raw):
                x_gt_scaled = int(x_gt_raw * (w / self.original_w))
                y_gt_scaled = int(y_gt_raw * (h / self.original_h))

                # 使用缩放后的坐标来绘制绿色的真实标记
                cv2.drawMarker(input_frame_bgr, (x_gt_scaled, y_gt_scaled), color=(0, 255, 0), 
                               markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

            # 绘制红色的预测标记
            if x_pred is not None:
                cv2.drawMarker(input_frame_bgr, (int(x_pred), int(y_pred)), color=(0, 0, 255), 
                               markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            
            # --- 拼接画布 ---
            canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
            canvas[:, 0:w] = input_frame_bgr
            canvas[:, w:2*w] = pred_heatmap_color
            canvas[:, 2*w:3*w] = target_heatmap_color
            
            cv2.putText(canvas, 'Input (Red:Pred, Green:GT)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(canvas, 'Prediction', (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(canvas, 'Ground Truth', (2*w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            # 保存图像
            save_path = self.vis_dir / f'sample_{self.vis_count}.jpg'
            cv2.imwrite(str(save_path), canvas)
            self.vis_count += 1