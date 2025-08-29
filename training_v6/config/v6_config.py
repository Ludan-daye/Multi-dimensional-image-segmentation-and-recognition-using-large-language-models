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
    output_attentions: bool = True
    # 分类策略
    cls_strategy: str = "last_non_pad"  # 可选: explicit, last_non_pad, global_pool_mean, attn_pool
    pad_token_id: int = 0
    eos_token_id: int = 5
    cls_token_id: int = 4


@dataclass
class DataConfig:
    train_data_path: str = "training_v3/generated_sequences_super_enhanced/sequences_labels_fixed_v2_maxlen1024.jsonl"
    vocab_path: str = "training_v3/outputs/best_model_silent/vocab.json"
    max_length: int = 1024
    num_classes: int = 10
    seed: int = 42


@dataclass
class TrainingConfig:
    num_epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    classification_loss_weight: float = 1.2
    lm_loss_weight: float = 0.03
    num_workers: int = 2
    use_weighted_sampler: bool = False
    enable_attention_supervision: bool = True
    attention_loss_weight: float = 0.01
    warmup_ratio: float = 0.2
    # 学习率与调度
    head_lr_mult: float = 1.0
    scheduler_type: str = "warmup_cosine"  # 可选: warmup_cosine, cosine_restarts
    cosine_restart_T0: int = 20            # 仅在 cosine_restarts 下生效（单位：epoch）
    # 训练控制
    resume: bool = True
    write_pid: bool = True
    run_name: str = "default"
    # 梯度裁剪
    max_grad_norm: float = 0.0
    # 评估输出
    save_confusion_matrix: bool = True
    outputs_dir: str = "training_v6/outputs"


@dataclass
class V6Config:
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


def get_v6_config() -> V6Config:
    return V6Config()


