# --- 1. 触发所有模块的“自我注册” ---
# 导入我们定义的数据集模块。
# 这个动作本身会执行 tracknet_dataset.py 文件，从而运行里面的
# @DATASETS.register_module 装饰器，完成注册。
from .datasets import tracknet_dataset
from  .datasets import tracknetv2_dataset

# 导入我们定义的所有自定义transform模块。
# 同理，这将触发所有 @TRANSFORMS.register_module 的执行。
# from .transforms import utracknetv1_transforms
from .transforms import tracknetv2_transforms

# --- 2. 提供简洁的外部接口 ---
# 从我们内部的 builder.py 文件中，将核心的构建函数“提升”到工厂的“门面”上。
from .builder import build_dataset, build_pipeline


# --- 3. (好习惯) 定义包的公共API ---
# __all__ 告诉其他程序，当使用 from datasets_factory import * 时，
# 应该导入哪些名字。我们只暴露用户需要直接调用的高级函数。
__all__ = [
    'build_dataset', 'build_pipeline'
]