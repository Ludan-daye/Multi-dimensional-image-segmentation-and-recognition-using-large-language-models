#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V5 配置：数据、模型与训练参数
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any
import json
from pathlib import Path


@dataclass
class ModelConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_positions: int = 1024
    vocab_size: int = 516
    dropout: float = 0.1
    # 分类头相关
    use_multi_layer_pool: bool = True  # 使用多层特征聚合（最近K层平均）
    pool_last_k: int = 4               # 聚合的最近K层
    use_mlp_head: bool = True          # 使用MLP分类头，否则为线性
    mlp_hidden: int = 512              # MLP隐藏维度
    head_dropout: float = 0.1          # 分类头dropout


@dataclass
class DataConfig:
    train_data_path: str = "training_v3/generated_sequences_super_enhanced/sequences_labels_fixed_v2_maxlen1024.jsonl"
    vocab_path: str = "training_v3/outputs/best_model_silent/vocab.json"
    max_length: int = 1024
    num_classes: int = 10
    seed: int = 42


@dataclass
class TrainingConfig:
    num_epochs: int = 500
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    classification_loss_weight: float = 0.9
    lm_loss_weight: float = 0.1
    num_workers: int = 2
    use_weighted_sampler: bool = True
    resume: bool = True              # 是否从latest检查点恢复
    # 优化分类头的训练
    head_lr_mult: float = 5.0          # 分类头学习率倍率
    label_smoothing: float = 0.1       # 标签平滑（CrossEntropyLoss）
    freeze_backbone_epochs: int = 0    # 训练最初N个epoch冻结骨干
    max_grad_norm: float = 1.0         # 梯度裁剪阈值
    # Focal Loss
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.0           # 0表示不使用alpha调权


@dataclass
class V5Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': asdict(self.model),
                'data': asdict(self.data),
                'training': asdict(self.training),
            }, f, ensure_ascii=False, indent=2)


def get_v5_config() -> V5Config:
    return V5Config()


