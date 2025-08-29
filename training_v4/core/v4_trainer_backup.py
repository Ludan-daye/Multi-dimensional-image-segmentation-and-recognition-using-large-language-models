#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training V4 终极训练器 - 完整版
严格按照用户所有要求实现：
1. 跑满GPU - 混合精度+数据并行+优化批次
2. 静默训练 - 后台运行+日志重定向
3. 断点可恢复 - 自动保存检查点
4. 终端关闭可持续 - 信号处理
5. 随时输出进度 - 多线程监控
6. 训练逻辑正确 - 明确分类目标
7. 1024全局长度 - 保护CLS token
"""

import os
import sys
import json
import time
import signal
import logging
import traceback
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2LMHeadModel, GPT2Config
from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.append('.')
sys.path.append('..')

from config.advanced_config import get_v4_config
from core.enhanced_data_processor import EnhancedDataProcessor, ImageClassificationDataset
from core.training_objectives import TrainingObjective

class V4Trainer:
    """V4终极训练器 - 满足用户所有要求"""
    
    def __init__(self):
        print("🚀 初始化V4终极训练器...")
        print("=" * 60)
        print("✅ 满足用户要求:")
        print("   1. 跑满GPU - 混合精度+数据并行+优化批次")
        print("   2. 静默训练 - 后台运行+日志输出")
        print("   3. 断点可恢复 - 自动保存检查点")
        print("   4. 终端关闭可持续 - nohup支持")
        print("   5. 随时输出进度 - 多线程监控")
        print("   6. 训练逻辑正确 - 明确分类目标")
        print("   7. 1024全局长度 - 保护CLS token")
        print("=" * 60)
        
        # 基础配置
        self.config = get_v4_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置日志系统
        self._setup_logging()
        
        # 核心组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler()
        self.training_objective = TrainingObjective()
        self.data_processor = None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.training_start_time = None
        
        # 监控系统
        self.checkpoint_dir = Path("outputs/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.step_times = []
        self.gpu_stats = []
        
        # 多线程控制
        self.stop_monitoring = False
        self.progress_thread = None
        
        # 训练目标映射 - 确保理解正确
        self.cls_token_to_label = {}
        self.label_to_cls_token = {}
        
        self.logger.info("✅ V4训练器初始化完成")
        self.logger.info(f"🎯 设备: {self.device}")
        self.logger.info(f"📊 模型配置: {self.config.model.n_embd}d-{self.config.model.n_layer}层")
        self.logger.info(f"📏 序列长度: {self.config.model.n_positions}")
    
    def _setup_logging(self):
        """设置完整的日志系统"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"v4_training_{timestamp}.log"
        
        # 配置日志处理器
        handlers = [logging.FileHandler(self.log_file, encoding='utf-8')]
        
        # 如果是终端模式，也输出到控制台
        if sys.stdout.isatty():
            handlers.append(logging.StreamHandler(sys.stdout))
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)8s | %(message)s',
            handlers=handlers,
            force=True
        )
        
        self.logger = logging.getLogger(__name__)
        
        # 静默模式处理
        if not sys.stdout.isatty():
            sys.stdout = open(self.log_file, 'a', encoding='utf-8')
            sys.stderr = open(self.log_file, 'a', encoding='utf-8')
            self.logger.info("🔇 静默模式激活")
        
        self.logger.info(f"📝 日志文件: {self.log_file}")
    
    def _setup_classification_mapping(self, vocab: Dict[str, int]):
        """建立分类映射 - 确保训练目标完全正确"""
        self.logger.info("🎯 建立分类目标映射...")
        
        self.cls_token_to_label.clear()
        self.label_to_cls_token.clear()
        
        for label in range(10):
            cls_token = f"<CLS_{label}>"
            if cls_token in vocab:
                token_id = vocab[cls_token]
                self.label_to_cls_token[label] = token_id
                self.cls_token_to_label[token_id] = label
            else:
                self.logger.error(f"❌ 找不到分类token: {cls_token}")
        
        if len(self.label_to_cls_token) != 10:
            self.logger.error(f"❌ 分类映射不完整! 只找到 {len(self.label_to_cls_token)}/10 个分类token")
            raise ValueError("分类token映射不完整")
        
        self.logger.info(f"✅ 分类映射建立完成: {len(self.label_to_cls_token)} 个类别")
        self.logger.info("🎯 训练目标明确: 在<CLS>位置预测正确的<CLS_X>分类token")
    
    def _validate_training_sample(self, batch, batch_idx: int):
        """验证训练样本的正确性"""
        if batch_idx % 500 == 0:
            self.logger.info("🔍 验证训练样本一致性...")
            
            sample_count = min(2, len(batch['cls_label']))
            for i in range(sample_count):
                true_label = batch['cls_label'][i].item()
                target_token = batch['cls_target_token'][i].item()
                expected_token = self.label_to_cls_token.get(true_label)
                cls_pos = batch['cls_position'][i]
                
                if target_token != expected_token:
                    self.logger.error(f"❌ 样本{i}训练目标不一致!")
                    raise ValueError("训练数据不一致")
                else:
                    self.logger.debug(f"✅ 样本{i}目标正确: 标签{true_label} -> token {target_token}")
    
    def _initialize_model(self):
        """初始化模型 - 充分利用GPU"""
        self.logger.info("🧠 初始化GPT2模型...")
        
        model_config = GPT2Config(**self.config.get_model_config_dict())
        self.model = GPT2LMHeadModel(model_config)
        self.model.to(self.device)
        
        # 多GPU并行
        if torch.cuda.device_count() > 1:
            self.logger.info(f"🚀 使用 {torch.cuda.device_count()} 个GPU进行并行训练")
            self.model = nn.DataParallel(self.model)
        
        # 统计模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"📊 模型参数: {total_params:,} ({total_params * 4 / (1024**3):.2f} GB)")
        
        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        total_steps = self.config.training.num_epochs * 1000
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.training.learning_rate,
            total_steps=total_steps,
            pct_start=0.1
        )
        
        self.logger.info("✅ 模型初始化完成")
    
    def _load_and_prepare_data(self):
        """加载并准备数据 - 1024全局长度"""
        self.logger.info("📁 加载和准备训练数据...")
        
        # 加载词汇表
        vocab_path = self.config.data.vocab_path
        if vocab_path == "默认词汇表":
            vocab = self.config.vocab
        else:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
        
        self.logger.info(f"📊 词汇表大小: {len(vocab)} tokens")
        
        # 建立分类映射
        self._setup_classification_mapping(vocab)
        
        # 初始化数据处理器
        self.data_processor = EnhancedDataProcessor(
            vocab=vocab,
            max_length=self.config.model.n_positions
        )
        
        # 加载和处理数据
        processed_data = self.data_processor.load_and_process_data(
            self.config.data.train_data_path
        )
        
        # 简单数据划分
        total_size = len(processed_data)
        train_size = int(total_size * 0.8)
        val_size = total_size - train_size
        
        train_data = processed_data[:train_size]
        val_data = processed_data[train_size:]
        
        # 创建数据集和加载器
        train_dataset = ImageClassificationDataset(train_data)
        val_dataset = ImageClassificationDataset(val_data)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.logger.info(f"📊 数据加载完成: 训练{len(self.train_loader)}批次, 验证{len(self.val_loader)}批次")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点 - 支持断点恢复"""
        try:
            model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
            
            checkpoint = {
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_accuracy': self.best_accuracy,
                'cls_mapping': {
                    'label_to_token': self.label_to_cls_token,
                    'token_to_label': self.cls_token_to_label
                },
                'timestamp': datetime.now().isoformat()
            }
            
            latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
            torch.save(checkpoint, latest_path)
            
            if is_best:
                best_path = self.checkpoint_dir / "best_model_v4.pt"
                torch.save(checkpoint, best_path)
                self.logger.info(f"🏆 保存最佳模型: accuracy={self.best_accuracy:.4f}")
            
            self.logger.info(f"💾 检查点保存: epoch={epoch}, step={self.global_step}")
            
        except Exception as e:
            self.logger.error(f"❌ 检查点保存失败: {e}")
    
    def _load_checkpoint(self) -> bool:
        """加载检查点 - 断点恢复"""
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        
        if not latest_path.exists():
            self.logger.info("📁 未找到检查点，从头开始训练")
            return False
        
        try:
            self.logger.info(f"🔄 加载检查点: {latest_path}")
            checkpoint = torch.load(latest_path, map_location=self.device)
            
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_accuracy = checkpoint['best_accuracy']
            
            if 'cls_mapping' in checkpoint:
                self.label_to_cls_token = checkpoint['cls_mapping']['label_to_token']
                self.cls_token_to_label = checkpoint['cls_mapping']['token_to_label']
            
            self.logger.info(f"✅ 检查点加载成功: epoch={self.current_epoch}, 最佳准确率={self.best_accuracy:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 检查点加载失败: {e}")
            return False
    
    def _start_monitoring(self):
        """启动监控线程"""
        self.progress_thread = threading.Thread(target=self._progress_monitor, daemon=True)
        self.progress_thread.start()
        self.logger.info("✅ 监控线程已启动")
    
    def _progress_monitor(self):
        """进度监控线程"""
        while not self.stop_monitoring:
            try:
                if self.global_step > 0 and self.training_start_time:
                    elapsed = time.time() - self.training_start_time
                    steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0
                    
                    total_steps = self.config.training.num_epochs * len(self.train_loader)
                    remaining_steps = total_steps - self.global_step
                    eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                    eta = str(timedelta(seconds=int(eta_seconds)))
                    
                    progress_msg = (
                        f"🔄 V4训练进度 - Epoch: {self.current_epoch}/{self.config.training.num_epochs} | "
                        f"Step: {self.global_step}/{total_steps} | Speed: {steps_per_sec:.2f} steps/s | "
                        f"ETA: {eta} | Best Acc: {self.best_accuracy:.4f}"
                    )
                    
                    self.logger.info(progress_msg)
                
                time.sleep(60)  # 每分钟输出一次
                
            except Exception as e:
                self.logger.error(f"❌ 进度监控异常: {e}")
                time.sleep(30)
    
    def _train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        classification_correct = 0
        total_valid_samples = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}",
            leave=False,
            disable=not sys.stdout.isatty()
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 验证训练样本
            self._validate_training_sample(batch, batch_idx)
            
            # 数据准备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            cls_positions = batch['cls_position']
            cls_targets = batch['cls_target_token'].to(self.device)
            cls_labels = batch['cls_label'].to(self.device)
            
            # 前向传播
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # 使用训练目标类计算损失
                total_loss_batch, loss_components = self.training_objective.compute_loss(
                    outputs.logits, outputs.loss, cls_positions, cls_targets, cls_labels
                )
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss_batch).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # 统计
            total_loss += loss_components['total_loss']
            classification_correct += loss_components['classification_accuracy'] * loss_components['valid_samples']
            total_valid_samples += loss_components['valid_samples']
            self.global_step += 1
            
            # 更新进度条
            if sys.stdout.isatty():
                current_acc = classification_correct / max(total_valid_samples, 1)
                progress_bar.set_postfix({
                    'Loss': f'{loss_components["total_loss"]:.4f}',
                    'Acc': f'{current_acc:.4f}',
                    'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
                })
            
            # 定期保存和验证
            if self.global_step % 500 == 0:
                self._save_checkpoint(epoch)
            
            if self.global_step % 100 == 0:
                val_accuracy = self._validate()
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    self._save_checkpoint(epoch, is_best=True)
                self.model.train()
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = classification_correct / max(total_valid_samples, 1)
        
        self.logger.info(f"📊 Epoch {epoch+1} 结果: 损失={epoch_loss:.4f}, 准确率={epoch_accuracy:.4f}")
        return epoch_loss, epoch_accuracy
    
    def _validate(self) -> float:
        """验证模型"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                cls_positions = batch['cls_position']
                cls_targets = batch['cls_target_token'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                for i, cls_pos in enumerate(cls_positions):
                    if cls_pos != -1:
                        cls_logits = logits[i, cls_pos, :]
                        pred_token = torch.argmax(cls_logits)
                        if pred_token == cls_targets[i]:
                            total_correct += 1
                        total_samples += 1
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        self.logger.info(f"🎯 验证准确率: {accuracy:.4f} ({total_correct}/{total_samples})")
        return accuracy
    
    def train(self):
        """主训练循环"""
        self.logger.info("🚀 开始V4训练...")
        self.logger.info("=" * 80)
        print(self.training_objective.get_objective_summary())
        self.logger.info("=" * 80)
        
        try:
            self.training_start_time = time.time()
            
            # 初始化
            self._initialize_model()
            self._load_and_prepare_data()
            
            # 尝试恢复
            resumed = self._load_checkpoint()
            if resumed:
                self.logger.info("🔄 从检查点恢复训练")
            
            # 启动监控
            self._start_monitoring()
            
            # 训练循环
            for epoch in range(self.current_epoch, self.config.training.num_epochs):
                self.current_epoch = epoch
                self.logger.info(f"\n🚀 开始 Epoch {epoch+1}/{self.config.training.num_epochs}")
                
                train_loss, train_acc = self._train_epoch(epoch)
                val_accuracy = self._validate()
                
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self._save_checkpoint(epoch)
                
                self.logger.info(f"✅ Epoch {epoch+1} 完成 | 最佳准确率: {self.best_accuracy:.4f}")
            
            # 训练完成
            total_time = time.time() - self.training_start_time
            self.logger.info("\n" + "="*80)
            self.logger.info("🎉 V4训练完成!")
            self.logger.info(f"📊 总时间: {str(timedelta(seconds=int(total_time)))}")
            self.logger.info(f"📊 最佳准确率: {self.best_accuracy:.4f}")
            self.logger.info(f"📊 总步数: {self.global_step}")
            self.logger.info("=" * 80)
            
        except KeyboardInterrupt:
            self.logger.info("\n⚠️ 训练被中断")
            self._save_checkpoint(self.current_epoch)
        except Exception as e:
            self.logger.error(f"\n❌ 训练错误: {e}")
            self.logger.error(traceback.format_exc())
            self._save_checkpoint(self.current_epoch)
            raise
        finally:
            self.stop_monitoring = True

def setup_signal_handlers(trainer):
    """设置信号处理"""
    def signal_handler(signum, frame):
        print(f"\n🛑 接收到信号 {signum}，优雅停止...")
        trainer.stop_monitoring = True
        trainer._save_checkpoint(trainer.current_epoch)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """主函数"""
    print("🚀 启动Training V4 - 终极训练系统")
    print("=" * 80)
    print("✅ 按照用户要求实现:")
    print("   1. 跑满GPU - 混合精度+数据并行+优化批次")
    print("   2. 静默训练 - 后台运行+日志重定向")
    print("   3. 断点可恢复 - 自动保存检查点")
    print("   4. 终端关闭可持续 - 信号处理")
    print("   5. 随时输出进度 - 多线程监控")
    print("   6. 训练逻辑正确 - 明确分类目标")
    print("   7. 1024全局长度 - 保护CLS token")
    print("=" * 80)
    
    try:
        trainer = V4Trainer()
        setup_signal_handlers(trainer)
        
        print(f"\n🎯 训练配置确认:")
        print(f"   - 序列长度: {trainer.config.model.n_positions}")
        print(f"   - 批次大小: {trainer.config.training.batch_size}")
        print(f"   - 目标准确率: >20%")
        
        trainer.train()
        return 0
        
    except Exception as e:
        print(f"❌ 训练启动失败: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
