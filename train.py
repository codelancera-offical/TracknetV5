import argparse
import torch
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
import importlib.util
from pathlib import Path

# --- 1. 导入我们所有的“工厂”的建造函数 ---
# 导入顶层包，对应的 __init__.py 文件会确保所有模块都已注册
import models_factory
import datasets_factory
import losses_factory
import optimizers_factory
import metrics_factory
import engine

# 导入我们最终的“执行器” Runner
from engine.runner import Runner

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

def main():
    # --- A. 解析命令行参数并加载配置 ---
    # parser = argparse.ArgumentParser(description="A Unified Training Framework for TrackNet")
    # parser.add_argument('config', help='Path to the configuration file.')
    # args = parser.parse_args()
    #
    # cfg = load_config_from_path(args.config)
    cfg = load_config_from_path('./configs/experiments/tracknetv3_tennis_b2e500.py')
    print("✅ Configuration loaded successfully.")
    
    # --- B. 环境设置 (由 Runner 内部处理或在这里设置) ---
    # 为了让 Runner 更独立，我们将 device, seed, work_dir 的创建都交给了 Runner
    
    # --- C. 使用工厂按图索骥，构建所有组件 ---
    print("Building components from config...")
    
    # 构建模型
    model = models_factory.build_model(cfg.model)
    print("✅ Model built successfully.")

    # 构建数据集
    train_dataset = datasets_factory.build_dataset(cfg.data['train'])
    val_dataset = datasets_factory.build_dataset(cfg.data['val'])
    print("✅ Datasets built successfully.")
    
    # 构建数据加载器 (DataLoader)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.data['samples_per_gpu'],
        num_workers=cfg.data['workers_per_gpu'],
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.data['samples_per_gpu'],
        num_workers=cfg.data['workers_per_gpu'],
        shuffle=False,
        pin_memory=True
    )
    print("✅ DataLoaders built successfully.")

    # 构建损失函数
    criterion = losses_factory.build_loss(cfg.loss)
    print("✅ Loss function built successfully.")

    # 构建优化器 (注意，它需要 model.parameters())
    optimizer = optimizers_factory.build_optimizer(model, cfg.optimizer)
    print("✅ Optimizer built successfully.")
    
    # 构建评估指标
    metric = metrics_factory.build_metric(cfg.evaluation['metric'])
    print("✅ Metric built successfully.")

    # 构建钩子 (Hooks)
    hooks = engine.build_hooks(cfg.log_config['hooks'])
    print("✅ Log Hooks built successfully.")
    # 检查是否存在自定义钩子配置，如果存在，则构建并添加进来
    if hasattr(cfg, 'custom_hooks'):
        hooks.extend(engine.build_hooks(cfg.custom_hooks))
        print("✅ Custom Hooks built successfully.")
    print("✅ All Hooks built successfully.")

    # 构建学习率调度器 (增加一个判断，使其成为可选项)
    if hasattr(cfg, 'lr_config') and cfg.lr_config is not None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.lr_config['step'][0],
            gamma=cfg.lr_config.get('gamma', 0.1)
        )
        print("✅ LR scheduler built successfully.")
    else:
        lr_scheduler = None # 如果没有配置，则为 None
        print("ℹ️ No LR scheduler configured. Running with a fixed learning rate.")
    
    # --- D. 实例化“赛车手”(Runner) ---
    # 将所有构建好的组件“装备”给Runner
    runner = Runner(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metric=metric,
        train_loader=train_loader,
        val_loader=val_loader,
        lr_scheduler=lr_scheduler,
        hooks=hooks,
        cfg=cfg
    )
    
    # --- E. 启动训练！---
    runner.run()

if __name__ == '__main__':
    main()