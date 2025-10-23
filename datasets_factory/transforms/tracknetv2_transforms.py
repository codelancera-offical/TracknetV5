# 文件: datasets_factory/transforms/utracknetv1_transforms.py

import cv2
import numpy as np
import torch
from pathlib import Path

# 导入我们自己的TRANSFORMS注册表
from ..builder import TRANSFORMS


@TRANSFORMS.register_module
class LoadMultiImagesFromPaths:
    """从results字典中的文件路径加载多张图像。"""

    def __init__(self, to_rgb=True):
        self.to_rgb = to_rgb

    def __call__(self, results: dict) -> dict:
        # 'img_fields' 是我们在Dataset中准备好的、需要加载的图片键名列表
        for key in results['img_fields']:
            img_path = results[key]
            if not isinstance(img_path, Path) or not img_path.exists():
                raise FileNotFoundError(f"Image file not found at path: {img_path} for key: {key}")

            img = cv2.imread(str(img_path))
            if self.to_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results[key] = img
        return results


@TRANSFORMS.register_module
class Resize:
    """对字典中的多张图像进行缩放。"""

    def __init__(self, keys, size: tuple):
        # size a tuple of (height, width)
        self.keys = keys
        self.size_wh = (size[1], size[0])  # cv2.resize expects (width, height)

    def __call__(self, results: dict) -> dict:
        for key in self.keys:
            if key in results:
                results[key] = cv2.resize(results[key], self.size_wh)
        return results


@TRANSFORMS.register_module
class ConcatChannels:
    """按指定顺序拼接通道，形成最终的模型输入。"""

    def __init__(self, keys, output_key='image'):
        self.keys = keys
        self.output_key = output_key

    def __call__(self, results: dict) -> dict:
        imgs_to_stack = [results[key] for key in self.keys]
        results[self.output_key] = np.concatenate(imgs_to_stack, axis=2)
        return results


@TRANSFORMS.register_module
class LoadAndFormatMultiTargets:
    """加载、缩放并格式化多张GT热力图为Tensor，维度为[3, h, w]。"""

    def __init__(self, keys=['gt_path_prev', 'gt_path', 'gt_path_next'], output_key='target'):
        self.keys = keys
        self.output_key = output_key

    def __call__(self, results: dict) -> dict:
        targets = []
        size = (results['input_width'], results['input_height'])

        for key in self.keys:
            gt_path = results[key]
            target_np = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            target_np = cv2.resize(target_np, size, interpolation=cv2.INTER_NEAREST)
            targets.append(target_np)

        # 堆叠成 [3, H, W] 维度
        target_stack = np.stack(targets, axis=0)  # 形状: (3, H, W)
        results[self.output_key] = torch.from_numpy(target_stack.astype(np.float32))
        return results


@TRANSFORMS.register_module
class Finalize:
    """
    将数据转为Tensor并收集最终需要的键值对，作为dataloader的最终输出。
    """

    def __init__(self, image_key='image', final_keys=['image', 'target', 'coords', 'visibility']):
        self.image_key = image_key
        self.final_keys = final_keys

    def __call__(self, results: dict) -> dict:
        # 将最终的输入图像转为 PyTorch 需要的 (C, H, W) 格式 Tensor
        img = results[self.image_key]
        results[self.image_key] = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255)  # <-- 修正这里

        # 从“周转箱”中只挑选出模型训练/评估需要的最终数据
        final_data = {}
        for key in self.final_keys:
            if key in results:
                final_data[key] = results[key]
        return final_data