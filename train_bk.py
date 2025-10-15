# 文件路径: ./scripts/train_utracknet.py (已修正编码问题)

from engine.losses import get_criterion
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import yaml
import argparse
from pathlib import Path

# 导入我们重构好的所有模块
from datasets import TrackNetDataset
from archs import UTrackNetModel
from engine.core import train_one_epoch, validate

def main(config):
    # --- 1. 设置实验路径 ---
    exp_path = Path('./exps') / config['exp_id']
    exp_path.mkdir(parents=True, exist_ok=True)
    
    # --- 2. 加载数据 ---
    data_cfg = config['data']
    
    if data_cfg['use_motion_attention']:
        csv_name_train = f"labels_context_train.csv"
        csv_name_val = f"labels_context_val.csv"
    else:
        csv_name_train = f"labels_{data_cfg['mode']}_train.csv"
        csv_name_val = f"labels_{data_cfg['mode']}_val.csv"

    train_csv_path = Path(data_cfg['data_dir']) / csv_name_train
    val_csv_path = Path(data_cfg['data_dir']) / csv_name_val
    
    train_dataset = TrackNetDataset(
        data_dir=data_cfg['data_dir'],
        csv_path=train_csv_path,
        use_motion_attention=data_cfg['use_motion_attention'],
        input_height=data_cfg['height'],
        input_width=data_cfg['width']
    )
    
    dl_cfg = config['dataloader']
    train_loader = DataLoader(train_dataset, batch_size=dl_cfg['batch_size'], 
                              shuffle=dl_cfg['shuffle'], num_workers=dl_cfg['num_workers'],
                              pin_memory=dl_cfg['pin_memory'])

    val_dataset = TrackNetDataset(
        data_dir=data_cfg['data_dir'],
        csv_path=val_csv_path,
        use_motion_attention=data_cfg['use_motion_attention'],
        input_height=data_cfg['height'],
        input_width=data_cfg['width']
    )
    val_loader = DataLoader(val_dataset, batch_size=dl_cfg['batch_size'], shuffle=False,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
    
    # --- 3. 创建模型 ---
    in_channels = 13 if data_cfg['use_motion_attention'] else 9
    model_cfg = config['model']
    model = UTrackNetModel(in_channels=in_channels, num_classes=model_cfg['num_classes'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # --- 4. 设置优化器和日志 ---
    train_cfg = config['train']
    if train_cfg['optimizer'].lower() == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=train_cfg['lr'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'])
    
    criterion = get_criterion(config).to(device)

    writer = SummaryWriter(log_dir= str(exp_path / 'logs'))
    model_best_path = exp_path / 'model_best.pth'
    
    # --- 5. 开始训练循环 ---
    val_best_f1 = 0
    val_cfg = config['validation']

    for epoch in range(train_cfg['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion ,optimizer, device, epoch, train_cfg['steps_per_epoch'])
        writer.add_scalar('Train/Loss', train_loss, epoch)
        
        if (epoch + 1) % val_cfg['val_intervals'] == 0:
            val_loss, precision, recall, f1 = validate(model, val_loader, criterion,device, val_cfg['min_dist'])
            
            print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/Precision', precision, epoch)
            writer.add_scalar('Val/Recall', recall, epoch)
            writer.add_scalar('Val/F1-Score', f1, epoch)
            
            if f1 > val_best_f1:
                val_best_f1 = f1
                torch.save(model.state_dict(), model_best_path)
                print(f"🚀 New best model saved to {model_best_path} with F1-score: {f1:.4f}")
    
    writer.close()
    print("🎉 Training finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    
    # ✨✨✨ 这里是唯一的修正点 ✨✨✨
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    main(config)