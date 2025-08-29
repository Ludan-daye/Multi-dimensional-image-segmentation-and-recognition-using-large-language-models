#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training V4 训练目标定义 - 确保训练逻辑正确
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class TrainingObjective:
    """训练目标类 - 明确定义训练逻辑"""
    
    def __init__(self):
        self.primary_objective = "图像分类"
        self.task_description = "在<CLS>token位置预测正确的<CLS_X>分类token"
        self.classification_weight = 0.8
        self.sequence_weight = 0.2
        
        print("🎯 训练目标已明确:")
        print(f"   - 主要任务: {self.primary_objective}")
        print(f"   - 任务描述: {self.task_description}")
    
    def compute_loss(self, logits, sequence_loss, cls_positions, cls_targets, cls_labels):
        """计算综合损失"""
        classification_loss = 0
        classification_correct = 0
        valid_samples = 0
        
        for i, cls_pos in enumerate(cls_positions):
            if cls_pos != -1:
                cls_logits = logits[i, cls_pos, :]
                cls_target = cls_targets[i]
                cls_loss = nn.CrossEntropyLoss()(cls_logits.unsqueeze(0), cls_target.unsqueeze(0))
                classification_loss += cls_loss
                
                pred_token = torch.argmax(cls_logits)
                if pred_token == cls_target:
                    classification_correct += 1
                valid_samples += 1
        
        if valid_samples > 0:
            classification_loss = classification_loss / valid_samples
            classification_accuracy = classification_correct / valid_samples
            total_loss = self.sequence_weight * sequence_loss + self.classification_weight * classification_loss
        else:
            classification_loss = torch.tensor(0.0, device=logits.device)
            classification_accuracy = 0.0
            total_loss = sequence_loss
        
        loss_components = {
            'total_loss': total_loss.item(),
            'sequence_loss': sequence_loss.item(),
            'classification_loss': classification_loss.item() if isinstance(classification_loss, torch.Tensor) else classification_loss,
            'classification_accuracy': classification_accuracy,
            'valid_samples': valid_samples
        }
        
        return total_loss, loss_components
    
    def get_objective_summary(self):
        return f"""
🎯 Training V4 训练目标摘要
====================================
主要任务: {self.primary_objective}
任务描述: {self.task_description}
成功标准: 分类准确率 > 20%
====================================
"""
