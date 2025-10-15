import cv2
import torch
import numpy as np
from pathlib import Path

from .base_hook import BaseHook
from ..builder import HOOKS

@HOOKS.register_module
class ValidationVisualizerHook(BaseHook):
    """
    一个在验证阶段，用于将模型预测结果可视化的钩子。
    它会将 输入-预测-真值 对比图保存到实验目录下。
    """
    def __init__(self, num_samples_to_save=5):
        self.num_samples_to_save = num_samples_to_save
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
        
        # 将数据从GPU转到CPU并转为Numpy
        input_tensor = batch['image'].cpu() # (B, C, H, W)
        target_tensor = batch['target'].cpu() # (B, H, W)
        pred_tensor = torch.argmax(logits, dim=1).cpu() # (B, H, W)
        
        # 遍历这个批次中的每一张图
        for i in range(input_tensor.size(0)):
            if self.vis_count >= self.num_samples_to_save:
                break
            
            # --- 构建可视化图像 ---
            # 只取第一个输入帧用于显示
            input_frame_rgb = (input_tensor[i, 5:8, :, :].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            input_frame_bgr = cv2.cvtColor(input_frame_rgb, cv2.COLOR_RGB2BGR)

            pred_heatmap_color = cv2.applyColorMap(pred_tensor[i].numpy().astype(np.uint8), cv2.COLORMAP_JET)
            target_heatmap_color = cv2.applyColorMap(target_tensor[i].numpy().astype(np.uint8), cv2.COLORMAP_JET)
            
            # 将三张图横向拼接
            h, w, _ = input_frame_bgr.shape
            canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
            canvas[:, 0:w] = input_frame_bgr
            canvas[:, w:2*w] = pred_heatmap_color
            canvas[:, 2*w:3*w] = target_heatmap_color
            
            # 添加文字标签
            cv2.putText(canvas, 'Input', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(canvas, 'Prediction', (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(canvas, 'Ground Truth', (2*w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            # 保存图像
            save_path = self.vis_dir / f'sample_{self.vis_count}.jpg'
            cv2.imwrite(str(save_path), canvas)
            self.vis_count += 1