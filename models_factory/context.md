# 模块: models_factory

## 1. 模块目标 (Module Goal)

本模块是整个项目的**模型构造工厂**。

其核心使命是，根据外部传入的**配置文件(config)**，动态地、灵活地构建和组装出各种深度学习模型。

## 2. 核心设计思想 (Core Design Philosophy)

为了实现最大的灵活性和可扩展性，我们采用了一种“**乐高积木**”式的设计哲学。一个完整的模型被看作是一个作品，由不同类型的“积木”搭建而成。

#### A. 组件化 (The "Lego Bricks")

我们将一个复杂的视觉模型拆分为三个标准化的组件：

-   **`Backbone` (骨干网络):** 模型的根基，如同乐高的“底盘和车身”。它负责从输入图像中提取多层次的初始特征。
-   **`Neck` (颈部):** 连接 `Backbone` 和 `Head` 的中间件，如同乐高的“连接件和装饰件”。它负责对 `Backbone` 提取的特征进行融合、加强或变换。
-   **`Head` (头部):** 模型的输出部分，如同乐高的“驾驶舱和发射器”。它根据 `Neck` 处理后的特征，计算出最终的任务特定结果（如热力图、坐标等）。

#### B. 注册表机制 (The "Brick Catalog")

我们如何让工厂“认识”所有可用的积木？答案是**注册表机制**。

-   **`builder.py`** 文件中的 `Registry` 类就是我们的“零件目录”。我们为每种组件都创建了一个专属目录 (例如 `BACKBONES`, `NECKS`, `HEADS`)。
-   任何一个新的组件类（无论写在哪个文件），只要在类定义前加上对应的**装饰器**（例如 `@BACKBONES.register_module`），就会像签到一样，自动将自己的信息（类名和类本身）登记到对应的“目录”里。
-   这个机制使得整个工厂系统是**可插拔的 (pluggable)**。新增一个组件，不需要修改任何工厂的核心代码，只需要“注册”即可。

#### C. 工厂函数 (The "Builder")

**`builder.py`** 中的 `build_model`, `build_backbone` 等函数是工厂的“**构建机器人**”。

-   它们接收一个配置字典 `cfg` (例如 `cfg['backbone']`)。
-   机器人读取 `cfg` 中的 `type` 字段（例如 `type: 'UTrackNetV1Backbone'`）。
-   然后去对应的“零件目录”（例如 `BACKBONES`）中查找这个名字。
-   找到后，就用 `cfg` 中的其他参数作为积木的“规格”，实例化这个组件类。
-   最终，`build_model` 函数会将构建好的 `backbone`, `neck`, `head` 组装成一个完整的模型并返回。

## 3. 关键文件导览 (Key File Guide)

-   `builder.py`: **工厂的核心**。管理所有注册表和 `build_*` 构建逻辑。
-   `backbones/`: **“Backbone”零件仓库**。存放所有骨干网络组件。
-   `necks/`: **“Neck”零件仓库**。存放所有颈部组件。
-   `heads/`: **“Head”零件仓库**。存放所有头部组件。
-   `models/`: **模型总装图**。存放如何将 `backbone`, `neck`, `head` 等组件拼接成一个完整模型的“配方”。
-   `__init__.py`: **工厂的统一出口**。将最关键的 `build_model` 函数暴露给外部模块，简化调用。

## 4. "如何新增一个组件" 工作流 (Workflow: How to Add a New Component)

假设您要新增一个名为 `MyBackbone` 的骨干网络：

1.  **创建组件文件**: 在 `models_factory/backbones/` 目录下创建一个新文件，例如 `my_backbone.py`。
2.  **编写并注册组件**: 在 `my_backbone.py` 中编写您的代码，并使用装饰器来注册它。

    ```python
    from ..builder import BACKBONES
    import torch.nn as nn

    @BACKBONES.register_module
    class MyBackbone(nn.Module):
        # ... 你的代码逻辑 ...
        pass
    ```

3.  **在配置文件中调用**: 现在，您可以在您的 `config` 文件中直接通过名字来使用这个新组件了。

    ```python
    # 在你的 config.py 文件中
    model = dict(
        type='UTrackNetV1', # 或者其他总装模型
        backbone=dict(
            type='MyBackbone', # <-- 在这里使用你新注册的组件
            # ... MyBackbone 需要的其他参数 ...
        ),
        neck=dict(...),
        head=dict(...)
    )
    ```

## 5. 外部交互 (External Interactions)

-   **上游依赖 (Depends on):**
    -   `torch.nn`: 所有模型组件的基础。
    -   `dict`: 通过 Python 字典格式的配置文件进行驱动。
-   **下游消费 (Used by):**
    -   通常由项目的主训练脚本（如 `train.py`）调用，用于在训练开始前构建模型。

    ```python
    # 在 train.py 中的使用示例
    from models_factory import build_model
    from utils import load_config

    cfg = load_config('my_config.py')
    model = build_model(cfg.model)
    ```