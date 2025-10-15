class BaseHook:
    """所有钩子(Hook)的基类。"""

    def before_run(self, runner):
        """在训练开始前被调用。"""
        pass

    def after_run(self, runner):
        """在训练结束后被调用。"""
        pass

    def before_epoch(self, runner):
        """在每个训练epoch开始前被调用。"""
        pass

    def after_epoch(self, runner):
        """在每个训练epoch结束后被调用。"""
        pass

    def before_iter(self, runner):
        """在每个训练iteration(batch)开始前被调用。"""
        pass

    def after_iter(self, runner):
        """在每个训练iteration(batch)结束后被调用。"""
        pass
    
    # --- ✨✨✨ 新增以下验证阶段的钩子接口 ✨✨✨ ---

    def before_val_epoch(self, runner):
        """在每个验证epoch开始前被调用。"""
        pass

    def after_val_epoch(self, runner):
        """在每个验证epoch结束后被调用。"""
        pass

    def before_val_iter(self, runner):
        """在每个验证iteration(batch)开始前被调用。"""
        pass

    def after_val_iter(self, runner):
        """在每个验证iteration(batch)结束后被调用。"""
        pass