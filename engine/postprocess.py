# 文件路径: ./engine/postprocess.py
import cv2
import numpy as np

def heatmap_to_coords(feature_map, threshold=127):
    """
    使用霍夫圆变换将热力图转换为坐标点。
    """
    # 假设输入的feature_map是[H, W]的numpy数组，数值范围0-255
    heatmap = feature_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, 
                               param1=50, param2=2, minRadius=2, maxRadius=7)
    
    if circles is not None and len(circles) == 1:
        x = circles[0][0][0]
        y = circles[0][0][1]
        return x, y
        
    return None, None