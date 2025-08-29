#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training V4 高级配置管理系统
整合v3的所有改进，增加自动调优和智能配置
"""

import json
import os
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
import torch

@dataclass
class PerformanceConfig:
    """性能优化配置"""
    # GPU优化
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    
    # 内存优化
    max_memory_fraction: float = 0.85
    empty_cache_steps: int = 100
    checkpoint_gradient: bool = True
    
    # 训练优化
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # 学习率调度
    scheduler_type: str = "cosine_with_restarts"  # cosine, cosine_with_restarts, polynomial
    lr_restart_cycles: int = 3
    lr_decay_factor: float = 0.8

@dataclass
class DataConfig:
    """数据配置 - V4增强版"""
    # 基础配置
    train_data_path: str = "../training_v3/generated_sequences_super_enhanced/sequences_labels_fixed_v2_maxlen1024.jsonl"
    vocab_path: str = "../training_v3/outputs/best_model_silent/vocab.json"
    
    # 序列配置
    max_length: int = 1024
    adaptive_length: bool = True  # 自适应序列长度
    padding_strategy: str = "max_length"  # max_length, longest, do_not_pad
    
    # 数据增强
    enable_augmentation: bool = True
    augmentation_ratio: float = 0.2
    noise_ratio: float = 0.05
    
    # 数据划分
    train_ratio: float = 0.8
    val_ratio: float = 0.15
    test_ratio: float = 0.05
    random_seed: int = 42
    
    # 缓存配置
    enable_caching: bool = True
    cache_dir: str = "data/cache"
    preload_data: bool = True

@dataclass
class ModelConfig:
    """模型配置 - V4智能版"""
    # 基础架构
    model_type: str = "gpt2"
    vocab_size: int = 516
    
    # 可调参数 - 支持自动调优
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 16
    n_positions: int = 1024
    
    # 高级配置
    activation_function: str = "gelu_new"
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    
    # Dropout配置
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    summary_first_dropout: float = 0.1
    
    # 特殊配置
    use_cache: bool = True
    scale_attn_weights: bool = True
    reorder_and_upcast_attn: bool = False
    
    # 自动调优相关
    auto_scale_for_gpu: bool = True
    target_gpu_utilization: float = 0.85
    min_batch_size: int = 2
    max_batch_size: int = 32

@dataclass
class TrainingConfig:
    """训练配置 - V4自适应版"""
    # 基础训练参数
    num_epochs: int = 500
    batch_size: int = 8
    learning_rate: float = 3e-4
    
    # 自适应配置
    adaptive_batch_size: bool = True
    adaptive_learning_rate: bool = True
    early_stopping: bool = True
    patience: int = 20
    
    # 学习率策略
    warmup_steps: int = 500
    min_learning_rate: float = 1e-6
    lr_schedule_patience: int = 10
    lr_schedule_factor: float = 0.5
    
    # 验证和保存
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 5
    
    # 实验追踪
    experiment_name: str = "training_v4_advanced"
    run_name: str = None  # 自动生成
    track_metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss", "perplexity", "lr"])
    
    # 恢复训练
    resume_from_checkpoint: bool = True
    checkpoint_dir: str = "outputs/checkpoints"

@dataclass
class MonitoringConfig:
    """监控配置 - V4全面监控"""
    # 基础监控
    enable_wandb: bool = False  # 可选：Weights & Biases
    enable_tensorboard: bool = True
    enable_mlflow: bool = False
    
    # GPU监控
    monitor_gpu: bool = True
    gpu_log_interval: int = 30  # 秒
    
    # 性能监控
    monitor_memory: bool = True
    monitor_throughput: bool = True
    profile_training: bool = False
    
    # 预警系统
    enable_alerts: bool = True
    accuracy_threshold: float = 0.15  # 准确率阈值
    loss_spike_threshold: float = 2.0  # 损失突增阈值
    
    # 日志配置
    log_level: str = "INFO"
    save_detailed_logs: bool = True
    log_predictions: bool = True
    max_prediction_samples: int = 100

@dataclass
class SpecialTokensConfig:
    """特殊Token配置 - V4标准化"""
    # 基础special tokens
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    eos_token: str = "<EOS>"
    bos_token: str = "<BOS>"  # V4新增
    
    # 图像相关
    img_start_token: str = "<IMG>"
    img_end_token: str = "</IMG>"
    
    # 分类相关
    cls_token: str = "<CLS>"
    cls_tokens: List[str] = field(default_factory=lambda: [f"<CLS_{i}>" for i in range(10)])
    
    # V4新增：语义标记
    sep_token: str = "<SEP>"  # 分隔符
    mask_token: str = "<MASK>"  # 掩码token
    
    # Token IDs (自动从vocab加载)
    token_ids: Dict[str, int] = field(default_factory=dict)

class AdvancedConfigManager:
    """V4高级配置管理器"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.config_file = self.base_dir / "config" / "v4_config.json"
        
        # 初始化各模块配置
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.performance = PerformanceConfig()
        self.monitoring = MonitoringConfig()
        self.special_tokens = SpecialTokensConfig()
        
        # 系统信息
        self.system_info = self._detect_system()
        self.gpu_info = self._detect_gpu()
        
        # 加载词汇表和自动配置
        self.vocab = self._load_vocab()
        self._setup_special_tokens()
        self._auto_configure()
        
        print(f"✅ V4配置管理器初始化完成")
        print(f"🎯 系统: {self.system_info['platform']}")
        print(f"🚀 GPU: {self.gpu_info['name']} ({self.gpu_info['memory_gb']:.1f}GB)")
    
    def _detect_system(self) -> Dict[str, Any]:
        """检测系统信息"""
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "architecture": platform.architecture()[0]
        }
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """检测GPU信息"""
        if not torch.cuda.is_available():
            return {"available": False, "name": "CPU", "memory_gb": 0}
        
        gpu_props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": gpu_props.name,
            "memory_gb": gpu_props.total_memory / (1024**3),
            "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
            "multiprocessor_count": getattr(gpu_props, 'multiprocessor_count', 'unknown')
        }
    
    def _load_vocab(self) -> Dict[str, int]:
        """加载词汇表"""
        vocab_path = Path(self.data.vocab_path)
        if vocab_path.exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
                print(f"✅ 从 {vocab_path} 加载词汇表 ({len(vocab)} tokens)")
                return vocab
        else:
            print(f"⚠️ 词汇表文件不存在: {vocab_path}")
            return self._create_default_vocab()
    
    def _create_default_vocab(self) -> Dict[str, int]:
        """创建默认词汇表"""
        vocab = {}
        
        # 特殊tokens
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        vocab['<IMG>'] = 2
        vocab['</IMG>'] = 3
        vocab['<CLS>'] = 4
        vocab['<EOS>'] = 5
        vocab['<BOS>'] = 6  # V4新增
        vocab['<SEP>'] = 7  # V4新增
        vocab['<MASK>'] = 8  # V4新增
        
        # 图像tokens
        for i in range(500):
            vocab[f'<Z_{i:03d}>'] = 9 + i
        
        # 分类tokens
        for i in range(10):
            vocab[f'<CLS_{i}>'] = 509 + i
        
        print(f"📝 创建默认词汇表 ({len(vocab)} tokens)")
        return vocab
    
    def _setup_special_tokens(self):
        """设置特殊token的ID映射"""
        for token_name in ['pad_token', 'unk_token', 'eos_token', 'bos_token', 
                          'img_start_token', 'img_end_token', 'cls_token', 'sep_token', 'mask_token']:
            token_value = getattr(self.special_tokens, token_name)
            if token_value in self.vocab:
                self.special_tokens.token_ids[token_name] = self.vocab[token_value]
        
        # 分类tokens
        for i, cls_token in enumerate(self.special_tokens.cls_tokens):
            if cls_token in self.vocab:
                self.special_tokens.token_ids[f'cls_{i}'] = self.vocab[cls_token]
    
    def _auto_configure(self):
        """根据系统信息自动配置"""
        if self.model.auto_scale_for_gpu and self.gpu_info['available']:
            self._auto_scale_model()
        
        if self.training.adaptive_batch_size:
            self._auto_adjust_batch_size()
        
        # 生成运行名称
        if not self.training.run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.training.run_name = f"v4_{timestamp}_gpu{int(self.gpu_info['memory_gb'])}gb"
    
    def _auto_scale_model(self):
        """根据GPU自动调整模型规模"""
        gpu_memory = self.gpu_info['memory_gb']
        
        if gpu_memory >= 80:  # A100 等高端卡
            self.model.n_embd = 1024
            self.model.n_layer = 16
            self.model.n_head = 16
            self.training.batch_size = 16
            print("🚀 检测到高端GPU，使用Large配置")
        elif gpu_memory >= 24:  # RTX 4090等
            self.model.n_embd = 768
            self.model.n_layer = 12
            self.model.n_head = 16
            self.training.batch_size = 12
            print("💪 检测到高性能GPU，使用Medium配置")
        elif gpu_memory >= 12:  # RTX 3080等
            self.model.n_embd = 512
            self.model.n_layer = 8
            self.model.n_head = 8
            self.training.batch_size = 8
            print("⚡ 检测到中端GPU，使用Small配置")
        else:  # 低端GPU
            self.model.n_embd = 384
            self.model.n_layer = 6
            self.model.n_head = 6
            self.training.batch_size = 4
            print("🔧 检测到入门GPU，使用Tiny配置")
    
    def _auto_adjust_batch_size(self):
        """根据模型大小自动调整批次大小"""
        # 估算参数量
        param_count = self._estimate_parameters()
        param_millions = param_count / 1_000_000
        
        # 根据参数量和GPU内存调整批次大小
        if param_millions > 500:  # 大模型
            self.training.batch_size = max(2, min(8, self.training.batch_size))
        elif param_millions > 200:  # 中模型
            self.training.batch_size = max(4, min(16, self.training.batch_size))
        else:  # 小模型
            self.training.batch_size = max(8, min(32, self.training.batch_size))
        
        print(f"📊 模型参数: {param_millions:.1f}M, 调整批次大小: {self.training.batch_size}")
    
    def _estimate_parameters(self) -> int:
        """估算模型参数量"""
        vocab_size = self.model.vocab_size
        n_embd = self.model.n_embd
        n_layer = self.model.n_layer
        n_positions = self.model.n_positions
        
        # 嵌入层参数
        emb_params = vocab_size * n_embd + n_positions * n_embd
        
        # Transformer层参数
        attn_params = 4 * n_embd * n_embd  # Q,K,V,O
        ffn_params = 8 * n_embd * n_embd   # FFN (4x expansion)
        norm_params = 2 * n_embd           # LayerNorm
        
        layer_params = (attn_params + ffn_params + norm_params) * n_layer
        
        # 输出层
        output_params = vocab_size * n_embd
        
        return emb_params + layer_params + output_params
    
    def get_model_config_dict(self) -> Dict:
        """获取HuggingFace兼容的模型配置"""
        return {
            "architectures": ["GPT2LMHeadModel"],
            "model_type": self.model.model_type,
            "vocab_size": self.model.vocab_size,
            "n_positions": self.model.n_positions,
            "n_embd": self.model.n_embd,
            "n_layer": self.model.n_layer,
            "n_head": self.model.n_head,
            "activation_function": self.model.activation_function,
            "resid_pdrop": self.model.resid_pdrop,
            "embd_pdrop": self.model.embd_pdrop,
            "attn_pdrop": self.model.attn_pdrop,
            "layer_norm_epsilon": self.model.layer_norm_epsilon,
            "initializer_range": self.model.initializer_range,
            "summary_type": "cls_index",
            "summary_use_proj": True,
            "summary_activation": None,
            "summary_proj_to_labels": True,
            "summary_first_dropout": self.model.summary_first_dropout,
            "scale_attn_weights": self.model.scale_attn_weights,
            "use_cache": self.model.use_cache,
            "pad_token_id": self.special_tokens.token_ids.get('pad_token', 0),
            "eos_token_id": self.special_tokens.token_ids.get('eos_token', 5),
            "bos_token_id": self.special_tokens.token_ids.get('bos_token', 6),
        }
    
    def save_config(self, save_path: Optional[str] = None):
        """保存完整配置"""
        if save_path is None:
            save_path = self.config_file
        
        config_dict = {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "performance": asdict(self.performance),
            "monitoring": asdict(self.monitoring),
            "special_tokens": asdict(self.special_tokens),
            "system_info": self.system_info,
            "gpu_info": self.gpu_info,
            "vocab_size": len(self.vocab),
            "estimated_parameters": self._estimate_parameters(),
            "created_at": datetime.now().isoformat()
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"💾 配置已保存到: {save_path}")
    
    def load_config(self, config_path: str):
        """加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        self.model = ModelConfig(**config_dict['model'])
        self.training = TrainingConfig(**config_dict['training'])
        self.data = DataConfig(**config_dict['data'])
        self.performance = PerformanceConfig(**config_dict['performance'])
        self.monitoring = MonitoringConfig(**config_dict['monitoring'])
        self.special_tokens = SpecialTokensConfig(**config_dict['special_tokens'])
        
        print(f"📂 配置已加载: {config_path}")
    
    def print_summary(self):
        """打印配置摘要"""
        print("\n" + "="*80)
        print(f"🚀 Training V4 配置摘要 - {self.training.experiment_name}")
        print("="*80)
        
        print(f"🖥️  系统信息:")
        print(f"   - 平台: {self.system_info['platform']}")
        print(f"   - GPU: {self.gpu_info['name']} ({self.gpu_info['memory_gb']:.1f}GB)")
        print(f"   - 内存: {self.system_info['memory_gb']:.1f}GB")
        
        print(f"🧠 模型配置:")
        print(f"   - 架构: {self.model.n_embd}d-{self.model.n_layer}层-{self.model.n_head}头")
        print(f"   - 序列长度: {self.model.n_positions}")
        print(f"   - 词汇表: {self.model.vocab_size} tokens")
        print(f"   - 估算参数: {self._estimate_parameters()/1_000_000:.1f}M")
        
        print(f"🎯 训练配置:")
        print(f"   - 轮数: {self.training.num_epochs}")
        print(f"   - 批次大小: {self.training.batch_size}")
        print(f"   - 学习率: {self.training.learning_rate}")
        print(f"   - 运行名称: {self.training.run_name}")
        
        print(f"📊 性能优化:")
        print(f"   - 混合精度: {self.performance.mixed_precision}")
        print(f"   - 梯度累积: {self.performance.gradient_accumulation_steps}")
        print(f"   - 调度器: {self.performance.scheduler_type}")
        
        print(f"📈 监控配置:")
        print(f"   - GPU监控: {self.monitoring.monitor_gpu}")
        print(f"   - TensorBoard: {self.monitoring.enable_tensorboard}")
        print(f"   - 详细日志: {self.monitoring.save_detailed_logs}")

# 便捷函数
def get_v4_config(config_name: str = "auto") -> AdvancedConfigManager:
    """获取V4配置"""
    manager = AdvancedConfigManager()
    
    if config_name != "auto":
        # 预定义配置
        configs = {
            "tiny": {"n_embd": 256, "n_layer": 4, "n_head": 4},
            "small": {"n_embd": 384, "n_layer": 6, "n_head": 6},
            "medium": {"n_embd": 768, "n_layer": 12, "n_head": 16},
            "large": {"n_embd": 1024, "n_layer": 16, "n_head": 16},
            "xlarge": {"n_embd": 1536, "n_layer": 20, "n_head": 24}
        }
        
        if config_name in configs:
            config = configs[config_name]
            manager.model.n_embd = config["n_embd"]
            manager.model.n_layer = config["n_layer"]
            manager.model.n_head = config["n_head"]
            print(f"🎯 使用预定义配置: {config_name}")
    
    return manager

if __name__ == "__main__":
    # 测试配置管理器
    print("🧪 测试V4配置管理器")
    
    config = get_v4_config()
    config.print_summary()
    config.save_config()
    
    print("\n✅ V4配置系统测试完成！")
