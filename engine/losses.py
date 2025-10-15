# 文件路径: ./engine/losses.py

import torch.nn as nn


def get_criterion(config: dict):
    """
    损失函数工厂：根据config字典创建并返回一个损失函数实例。
    """
    loss_config = config['train']['loss']
    loss_name = loss_config['name']
    loss_params = loss_config['params']

    if loss_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(**loss_params)
    elif loss_name == 'WBCE':
        from losses import WBCE_Loss_FromLogits
        return WBCE_Loss_FromLogits(**loss_params)
    else:
        raise NotImplementedError(f"Loss function '{loss_name}' is not implemented.")