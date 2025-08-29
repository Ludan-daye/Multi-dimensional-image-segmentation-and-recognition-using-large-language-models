#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import time
import sys
import os
from typing import Dict, Any
from collections import Counter
from pathlib import Path
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2Config

from training_v5.config.v5_config import get_v5_config
from training_v5.core.data_processor_v5 import V5DataProcessor, V5Dataset
from training_v5.core.model_v5 import GPT2WithCLSHead


class V5Trainer:
    def __init__(self):
        self.cfg = get_v5_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler()
        self.checkpoint_dir = Path('training_v5/outputs/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_acc = 0.0
        # 日志
        self._setup_logging()

    def _setup_logging(self):
        logs_dir = Path('training_v5/logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        self.log_file = logs_dir / f'v5_training_{ts}.log'
        handlers = [logging.FileHandler(self.log_file, encoding='utf-8')]
        if sys.stdout.isatty():
            handlers.append(logging.StreamHandler(sys.stdout))
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', handlers=handlers, force=True)
        self.logger = logging.getLogger(__name__)

    def _load_vocab(self) -> Dict[str, int]:
        # 支持多路径回退
        candidates = [
            self.cfg.data.vocab_path,
            '../training_v3/outputs/best_model_fixed/vocab.json',
            '../training_v3/generated_sequences_super_enhanced/vocab.json',
            'training_v3/outputs/best_model_fixed/vocab.json',
            'training_v3/generated_sequences_super_enhanced/vocab.json',
        ]
        for p in candidates:
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    vocab = json.load(f)
                return vocab
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"未找到词表文件，尝试路径: {candidates}")

    def _build_model(self) -> nn.Module:
        mcfg = self.cfg.model
        config = GPT2Config(
            n_layer=mcfg.n_layer,
            n_head=mcfg.n_head,
            n_embd=mcfg.n_embd,
            n_positions=mcfg.n_positions,
            vocab_size=mcfg.vocab_size,
            resid_dropout=mcfg.dropout,
            embd_dropout=mcfg.dropout,
            attn_pdrop=mcfg.dropout,
            output_hidden_states=True,
        )
        model = GPT2WithCLSHead(
            config,
            num_classes=self.cfg.data.num_classes,
            head_config=self.cfg.model,
        )
        return model.to(self.device)

    def _make_loaders(self):
        vocab = self._load_vocab()
        proc = V5DataProcessor(vocab, max_length=self.cfg.data.max_length, num_classes=self.cfg.data.num_classes)
        data = proc.load_and_process(self.cfg.data.train_data_path)
        train_rows, val_rows = V5DataProcessor.stratified_split(data, seed=self.cfg.data.seed, train_ratio=0.8)
        train_ds = V5Dataset(train_rows)
        val_ds = V5Dataset(val_rows)

        if self.cfg.training.use_weighted_sampler:
            labels = [int(x['cls_label']) for x in train_rows]
            cnt = Counter(labels)
            total = sum(cnt.values())
            class_weight = {c: total / (len(cnt) * cnt[c]) for c in cnt}
            sample_weights = [class_weight[int(x['cls_label'])] for x in train_rows]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=self.cfg.training.batch_size, sampler=sampler, num_workers=self.cfg.training.num_workers, pin_memory=True, drop_last=True)
        else:
            train_loader = DataLoader(train_ds, batch_size=self.cfg.training.batch_size, shuffle=True, num_workers=self.cfg.training.num_workers, pin_memory=True, drop_last=True)

        val_loader = DataLoader(val_ds, batch_size=self.cfg.training.batch_size, shuffle=False, num_workers=self.cfg.training.num_workers, pin_memory=True)
        return train_loader, val_loader

    def _validate_training_sample(self, batch, batch_idx: int):
        """基本一致性校验：label 与 <CLS_label> token 对齐，CLS 位置在有效范围内"""
        try:
            if batch_idx % 200 == 0:
                # 构造 label->token 的映射
                vocab = self._load_vocab()
                label_to_cls = {i: vocab.get(f"<CLS_{i}>") for i in range(self.cfg.data.num_classes)}
                for i in range(min(2, len(batch['cls_label']))):
                    true_label = int(batch['cls_label'][i])
                    target_token = int(batch['cls_target_token'][i])
                    expected_token = label_to_cls.get(true_label)
                    cls_pos = int(batch['cls_position'][i])
                    if expected_token is None or target_token != expected_token:
                        raise ValueError("训练样本分类token不一致")
                    if cls_pos < 0 or cls_pos >= self.cfg.data.max_length:
                        raise ValueError("CLS位置越界")
        except Exception as e:
            self.logger.error(f"样本一致性校验失败: {e}")
            raise

    def _build_optim(self, model: nn.Module):
        base_lr = self.cfg.training.learning_rate
        head_lr = base_lr * getattr(self.cfg.training, 'head_lr_mult', 1.0)
        params = [
            {"params": model.get_backbone_params(), "lr": base_lr, "weight_decay": self.cfg.training.weight_decay},
            {"params": model.get_head_params(), "lr": head_lr, "weight_decay": self.cfg.training.weight_decay},
        ]
        opt = optim.AdamW(params)
        return opt

    def _build_sched(self, optimizer, total_steps: int):
        # 先 warmup 后 cosine
        warmup_steps = max(10, total_steps // 10)
        cosine_steps = max(1, total_steps - warmup_steps)

        class WarmupThenCosine(optim.lr_scheduler._LRScheduler):
            def __init__(self, opt, warmup, cosine_total, last_epoch=-1):
                self.warmup = warmup
                self.cosine_total = cosine_total
                super().__init__(opt, last_epoch)
            def get_lr(self):
                base_lrs = [g['initial_lr'] if 'initial_lr' in g else g['lr'] for g in self.optimizer.param_groups]
                step = self.last_epoch + 1
                scale = 1.0
                if step <= self.warmup:
                    scale = step / float(max(1, self.warmup))
                else:
                    t = min(step - self.warmup, self.cosine_total)
                    scale = 0.5 * (1 + math.cos(math.pi * t / float(max(1, self.cosine_total))))
                return [lr * scale for lr in base_lrs]

        # 设置 initial_lr，兼容自定义调度
        for g in optimizer.param_groups:
            if 'initial_lr' not in g:
                g['initial_lr'] = g['lr']
        return WarmupThenCosine(optimizer, warmup_steps, cosine_steps)

    def _compute_loss(self, outputs, cls_logits, batch):
        seq_loss = outputs.loss
        # 分类损失在10类上
        target_labels = batch['cls_label'].to(self.device)
        label_smoothing = float(getattr(self.cfg.training, 'label_smoothing', 0.0))
        if getattr(self.cfg.training, 'use_focal_loss', False):
            gamma = float(getattr(self.cfg.training, 'focal_gamma', 2.0))
            alpha = float(getattr(self.cfg.training, 'focal_alpha', 0.0))
            logp = nn.functional.log_softmax(cls_logits, dim=-1)
            p = torch.exp(logp)
            nll = nn.functional.nll_loss(logp, target_labels, reduction='none')
            focal_factor = (1 - p.gather(1, target_labels.unsqueeze(1)).squeeze(1)).pow(gamma)
            if alpha > 0.0:
                alpha_t = torch.ones_like(target_labels, dtype=torch.float32, device=cls_logits.device) * alpha
                nll = alpha_t * nll
            cls_loss = (focal_factor * nll).mean()
        else:
            ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            cls_loss = ce(cls_logits, target_labels)
        total = self.cfg.training.lm_loss_weight * seq_loss + self.cfg.training.classification_loss_weight * cls_loss
        return total, seq_loss.detach().item(), cls_loss.detach().item()

    def _validate(self, model: nn.Module, val_loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attn = batch['attention_mask'].to(self.device)
                cls_pos = batch['cls_position']
                labels = batch['cls_label'].to(self.device)
                outputs, cls_logits = model(input_ids=input_ids, attention_mask=attn, labels=batch['labels'].to(self.device), cls_positions=cls_pos)
                preds = torch.argmax(cls_logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total else 0.0

    def train_one_epoch(self, model, train_loader, optimizer, scheduler):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # 样本一致性校验（抽样）
            self._validate_training_sample(batch, batch_idx)
            input_ids = batch['input_ids'].to(self.device)
            attn = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            cls_pos = batch['cls_position']
            with autocast():
                outputs, cls_logits = model(input_ids=input_ids, attention_mask=attn, labels=labels, cls_positions=cls_pos)
                loss, seq_l, cls_l = self._compute_loss(outputs, cls_logits, batch)
            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            # 梯度裁剪（未缩放的梯度需要先unscale）
            try:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(self.cfg.training, 'max_grad_norm', 0.0) or 0.0)
            except Exception:
                pass
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()
            total_loss += loss.detach().item()
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                self.logger.info(f"Progress - batch {batch_idx+1}/{len(train_loader)} | loss={loss.detach().item():.4f}")
        return total_loss / max(1, len(train_loader))

    def run_sanity(self):
        model = self._build_model()
        train_loader, val_loader = self._make_loaders()
        total_steps = self.cfg.training.num_epochs * len(train_loader)
        optimizer = self._build_optim(model)
        scheduler = self._build_sched(optimizer, total_steps)
        # 只跑一次验证，确认可用
        val_acc = self._validate(model, val_loader)
        print(f"SANITY_VAL_ACC={val_acc:.4f}")
        return True

    def _save_checkpoint(self, model: nn.Module, epoch: int, is_best: bool = False, optimizer=None, scheduler=None):
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'best_acc': self.best_acc,
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
        }
        latest = self.checkpoint_dir / 'latest_v5.pt'
        torch.save(state, latest)
        if is_best:
            best = self.checkpoint_dir / 'best_v5.pt'
            torch.save(state, best)

    def _try_resume(self, model: nn.Module, optimizer=None, scheduler=None):
        latest = self.checkpoint_dir / 'latest_v5.pt'
        if latest.exists():
            ckpt = torch.load(latest, map_location=self.device)
            # 兼容不同分类头：优先尝试完整加载，失败则仅加载骨干
            try:
                model.load_state_dict(ckpt['model'])
            except Exception as e:
                self.logger.warning(f"Full model load failed, fallback to backbone-only: {e}")
                backbone_only = {k: v for k, v in ckpt['model'].items() if k.startswith('backbone.')}
                model.load_state_dict(backbone_only, strict=False)
            # 优化器/调度器尽量加载，不匹配则跳过
            if optimizer is not None and ckpt.get('optimizer') is not None:
                try:
                    optimizer.load_state_dict(ckpt['optimizer'])
                except Exception as e:
                    self.logger.warning(f"Skip optimizer resume due to mismatch: {e}")
            if scheduler is not None and ckpt.get('scheduler') is not None:
                try:
                    scheduler.load_state_dict(ckpt['scheduler'])
                except Exception as e:
                    self.logger.warning(f"Skip scheduler resume due to mismatch: {e}")
            self.best_acc = ckpt.get('best_acc', 0.0)
            start_epoch = ckpt.get('epoch', -1) + 1
            self.logger.info(f"Resumed from {latest} at epoch {start_epoch}, best_acc={self.best_acc:.4f}")
            return max(0, start_epoch)
        return 0

    def train(self):
        model = self._build_model()
        train_loader, val_loader = self._make_loaders()
        optimizer = self._build_optim(model)
        total_steps = self.cfg.training.num_epochs * len(train_loader)
        scheduler = self._build_sched(optimizer, total_steps)
        self.logger.info(f"V5 TRAIN start: epochs={self.cfg.training.num_epochs}, steps/epoch={len(train_loader)}")
        # 写PID
        try:
            pid_file = Path('training_v5/logs/v5_training.pid')
            pid_file.write_text(str(os.getpid()))
        except Exception:
            pass
        # 尝试断点恢复
        start_epoch = 0
        if getattr(self.cfg.training, 'resume', True):
            start_epoch = self._try_resume(model, optimizer, scheduler)
        start = time.time()
        for epoch in range(start_epoch, self.cfg.training.num_epochs):
            # 可选：前若干轮冻结骨干
            freeze_epochs = int(getattr(self.cfg.training, 'freeze_backbone_epochs', 0))
            if freeze_epochs > 0:
                if epoch < freeze_epochs:
                    for p in model.get_backbone_params():
                        p.requires_grad = False
                elif epoch == freeze_epochs:
                    for p in model.get_backbone_params():
                        p.requires_grad = True
            loss = self.train_one_epoch(model, train_loader, optimizer, scheduler)
            val_acc = self._validate(model, val_loader)
            self.logger.info(f"Epoch {epoch+1}/{self.cfg.training.num_epochs} | loss={loss:.4f} | val_acc={val_acc:.4f}")
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self._save_checkpoint(model, epoch, is_best=True, optimizer=optimizer, scheduler=scheduler)
            else:
                self._save_checkpoint(model, epoch, is_best=False, optimizer=optimizer, scheduler=scheduler)
        dur = time.time() - start
        self.logger.info(f"V5 TRAIN done in {dur/60:.2f} min | best_val_acc={self.best_acc:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['sanity','train'], default='sanity')
    t = V5Trainer()
    args = ap.parse_args()
    if args.mode == 'sanity':
        t.run_sanity()
    else:
        t.train()


if __name__ == '__main__':
    main()


