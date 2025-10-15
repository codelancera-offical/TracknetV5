# 文件路径: ./engine/core.py
import torch
import numpy as np
import torch.nn as nn
from scipy.spatial import distance
from tqdm import tqdm
from .postprocess import heatmap_to_coords

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, steps_per_epoch):
    model.train()
    losses = []
    
    # 使用tqdm来显示进度条
    progress_bar = tqdm(train_loader, total=steps_per_epoch, desc=f"Train Epoch {epoch}")
    
    for i, batch in enumerate(progress_bar):
        # 从字典中获取数据，代码可读性更高
        inputs = batch['image'].to(device)
        targets = batch['target'].to(device)
        # print(f"traget before LOSS: {targets.shape}")
        
        optimizer.zero_grad()
        logits = model(inputs)
        # print(f"logits before LOSS:{logits.shape}")
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        progress_bar.set_postfix(loss=np.mean(losses))
        
        if i >= steps_per_epoch - 1:
            break
            
    return np.mean(losses)

def validate(model, val_loader, criterion, device, min_dist):
    model.eval()
    losses = []
    tp, fp, tn, fn = 0, 0, 0, 0
    
    progress_bar = tqdm(val_loader, desc="Validating")
    
    with torch.no_grad():
        for batch in progress_bar:
            inputs = batch['image'].to(device)
            targets = batch['target'].to(device)
            coords_gt = batch['coords']
            visibility_gt = batch['visibility']
            
            logits = model(inputs)
            loss = criterion(logits, targets)
            losses.append(loss.item())
            
            # 评估指标计算
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            for i in range(len(predictions)):
                x_pred, y_pred = heatmap_to_coords(predictions[i])
                x_gt, y_gt = coords_gt[0][i].item(), coords_gt[1][i].item()
                vis = visibility_gt[i].item()
                
                if x_pred is not None:
                    if vis != 0:
                        dist = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dist < min_dist:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        fp += 1
                else: # No prediction
                    if vis != 0:
                        fn += 1
                    else:
                        tn += 1
            progress_bar.set_postfix(loss=np.mean(losses))
            
    eps = 1e-15
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    return np.mean(losses), precision, recall, f1