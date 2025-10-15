import cv2
import numpy as np
from scipy.spatial import distance
import torch

from ..builder import METRICS

# --- 将 heatmap_to_coords 作为此模块的私有辅助函数 ---
# 它不属于类，因为它不依赖于类的状态 (self)
# 基于检测圆来推测坐标
# def _heatmap_to_coords(heatmap: np.ndarray, threshold: int = 127):
#     """
#     使用霍夫圆变换将热力图转换为坐标点。
#     """
#     heatmap_uint8 = heatmap.astype(np.uint8)
#     _, binary_map = cv2.threshold(heatmap_uint8, threshold, 255, cv2.THRESH_BINARY)
    
#     circles = cv2.HoughCircles(binary_map, cv2.HOUGH_GRADIENT, dp=1, minDist=1, 
#                                param1=50, param2=2, minRadius=1, maxRadius=20)
    
#     if circles is not None and len(circles) == 1:
#         x = circles[0][0][0]
#         y = circles[0][0][1]
#         return x, y
    
        
#     return None, None


def _heatmap_to_coords(heatmap: np.ndarray, threshold: int = 127):
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
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
            
    return None, None


@METRICS.register_module
class UTrackNetV1Metric:
    """
    一个专为UTrackNetV1设计的、用于计算 F1, Precision, Recall 的计分员。
    它内部封装了从热力图到坐标的转换逻辑。
    """
    def __init__(self, min_dist: int = 10, heatmap_threshold: int = 127):
        self.min_dist = min_dist
        self.heatmap_threshold = heatmap_threshold
        self.reset()

    def reset(self):
        """清空计分板。"""
        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0

    def update(self, logits: torch.Tensor, batch: dict):
        """根据一个批次的数据，更新计分板。"""
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        coords_gt = batch['coords']
        visibility_gt = batch['visibility']

        for i in range(len(predictions)):
            # 直接调用本文件内的辅助函数
            x_pred, y_pred = _heatmap_to_coords(predictions[i], threshold=self.heatmap_threshold)
            
            x_gt, y_gt = coords_gt[0][i].item(), coords_gt[1][i].item()
            vis = visibility_gt[i].item()
            
            if x_pred is not None:
                if vis != 0:
                    dist = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                    if dist < self.min_dist:
                        self.tp += 1
                    else:
                        self.fp += 1
                else:
                    self.fp += 1
            else:
                if vis != 0:
                    self.fn += 1
                else:
                    self.tn += 1

    def compute(self) -> dict:
        """计算并返回最终的评估结果字典。"""
        eps = 1e-15
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        
        return {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }