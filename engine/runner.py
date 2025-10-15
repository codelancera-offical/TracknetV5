import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np

class Runner:
    """
    一个封装了完整训练/验证循环的执行器。
    它负责管理训练状态、执行训练循环、调用钩子、进行验证和保存模型。
    """
    def __init__(self, model, optimizer, criterion, metric,
                 train_loader, val_loader, lr_scheduler, 
                 hooks, cfg):
        
        # --- 核心组件 ---
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_scheduler = lr_scheduler
        self.hooks = hooks
        self.cfg = cfg
        
        # --- 环境与路径 ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.work_dir = Path(cfg.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True) # 确保工作目录存在
        
        # --- 训练过程中的状态变量 ---
        self.epoch = 0
        self.global_iter = 0
        self.inner_iter = 0
        self.max_epochs = cfg.total_epochs
        
        # ✨ 使用 hasattr 检查可选参数，使 Runner 更健壮
        self.max_iters_per_epoch = cfg.steps_per_epoch if hasattr(cfg, 'steps_per_epoch') else len(self.train_loader)

        self.outputs = {}      # 用于在钩子之间传递临时数据 (如loss, metrics)
        self.best_metric = 0.0 # 用于保存最佳模型的判断依据

    def call_hooks(self, event_name):
        """调用所有钩子中名为 event_name 的方法。"""
        for hook in self.hooks:
            getattr(hook, event_name)(self)

    def train_epoch(self):
        """执行一个完整的训练 epoch。"""
        self.model.train()
        progress_bar = tqdm(self.train_loader, total=self.max_iters_per_epoch, 
                            desc=f"Train Epoch {self.epoch + 1}/{self.max_epochs}")
                            
        for i, data_batch in enumerate(progress_bar):
            if i >= self.max_iters_per_epoch:
                break

            self.inner_iter = i
            self.call_hooks('before_iter')
            
            inputs = data_batch['image'].to(self.device)
            targets = data_batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()
            
            self.global_iter += 1
            self.outputs['loss'] = loss.item()
            self.outputs['batch_size'] = inputs.size(0)
            self.current_lr = self.optimizer.param_groups[0]['lr']

            self.call_hooks('after_iter')
            progress_bar.set_postfix(loss=loss.item())

    @torch.no_grad()
    def validate_epoch(self):
        """
        执行一个完整的验证 epoch，并正确地调用钩子，为可视化提供支持。
        """
        self.model.eval()
        self.metric.reset()
        
        # 1. 新增：广播“验证epoch开始”事件
        # 这会触发 ValidationVisualizerHook 的 before_val_epoch 方法
        self.call_hooks('before_val_epoch')

        val_losses = []
        progress_bar = tqdm(self.val_loader, desc=f"Validate Epoch {self.epoch + 1}")
        for i, data_batch in enumerate(progress_bar):
            self.inner_iter = i
            
            inputs = data_batch['image'].to(self.device)
            targets = data_batch['target'].to(self.device)
            logits = self.model(inputs)
            
            # 2. 新增：将当前批次的数据暂存到 outputs 中，供钩子访问
            self.outputs['val_batch'] = data_batch
            self.outputs['val_logits'] = logits

            # 计算损失和指标
            loss = self.criterion(logits, targets)
            val_losses.append(loss.item())
            self.metric.update(logits, data_batch)
            
            # 3. 新增：广播“验证iter结束”事件
            # 这是 ValidationVisualizerHook 工作的关键！
            self.call_hooks('after_val_iter')

        # 计算并打印最终结果
        eval_results = self.metric.compute()
        eval_results['loss'] = np.mean(val_losses)
        self.outputs['val_metrics'] = eval_results 
        print(f"Validation Results: {eval_results}")
        
        # 4. 新增：广播“验证epoch结束”事件
        self.call_hooks('after_val_epoch')

    def run(self):
        """启动完整的训练流程。"""
        print("🚀 Starting Runner...")
        self.call_hooks('before_run')
        
        for self.epoch in range(self.max_epochs):
            self.call_hooks('before_epoch')
            self.train_epoch()
            
            if (self.epoch + 1) % self.cfg.evaluation['interval'] == 0:
                self.validate_epoch()
            
            # 保存最佳模型
            current_f1 = self.outputs.get('val_metrics', {}).get('F1-Score', 0.0)
            if current_f1 > self.best_metric:
                self.best_metric = current_f1
                best_model_path = self.work_dir / 'best_model.pth'
                torch.save(self.model.state_dict(), best_model_path)
                print(f"🏆 New best model saved to {best_model_path} with F1-score: {self.best_metric:.4f}")
            
            # 只有在学习率调度器存在时，才执行 .step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.call_hooks('after_epoch')
            
        self.call_hooks('after_run')
        print("\n🎉 Training finished!")
        print(f"Best F1-Score on validation set: {self.best_metric:.4f}")