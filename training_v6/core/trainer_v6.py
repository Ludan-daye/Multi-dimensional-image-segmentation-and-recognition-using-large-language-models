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

from training_v6.config.v6_config import get_v6_config
from training_v6.core.data_processor_v6 import V6DataProcessor, V6Dataset
from training_v6.core.model_v6 import GPT2WithCLSHead


class V6Trainer:
    def __init__(self):
        self.cfg = get_v6_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler()
        self.checkpoint_dir = Path('training_v6/outputs/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_acc = 0.0
        self._setup_logging()

    def _setup_logging(self):
        logs_dir = Path('training_v6/logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        self.log_file = logs_dir / f'v6_training_{ts}.log'
        handlers = [logging.FileHandler(self.log_file, encoding='utf-8')]
        if sys.stdout.isatty():
            handlers.append(logging.StreamHandler(sys.stdout))
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', handlers=handlers, force=True)
        self.logger = logging.getLogger(__name__)

    def _load_vocab(self) -> Dict[str, int]:
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
            output_attentions=self.cfg.model.output_attentions,
        )
        model = GPT2WithCLSHead(config, num_classes=self.cfg.data.num_classes, head_config=self.cfg.model)
        return model.to(self.device)

    def _make_loaders(self):
        vocab = self._load_vocab()
        proc = V6DataProcessor(vocab, max_length=self.cfg.data.max_length, num_classes=self.cfg.data.num_classes)
        data = proc.load_and_process(self.cfg.data.train_data_path)
        train_rows, val_rows = V6DataProcessor.stratified_split(data, seed=self.cfg.data.seed, train_ratio=0.8)
        train_ds = V6Dataset(train_rows)
        val_ds = V6Dataset(val_rows)
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
        try:
            if batch_idx % 200 == 0:
                vocab = self._load_vocab()
                label_to_cls = {i: vocab.get("<CLS_{}>".format(i)) for i in range(self.cfg.data.num_classes)}
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
        # 按分类头与主干分组，应用 head_lr_mult
        head_params = []
        base_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('cls_head') or 'cls_head' in name:
                head_params.append(p)
            else:
                base_params.append(p)
        head_lr_mult = float(getattr(self.cfg.training, 'head_lr_mult', 1.0))
        param_groups = [
            { 'params': base_params, 'lr': self.cfg.training.learning_rate },
            { 'params': head_params, 'lr': self.cfg.training.learning_rate * head_lr_mult },
        ]
        opt = optim.AdamW(param_groups, lr=self.cfg.training.learning_rate, weight_decay=self.cfg.training.weight_decay)
        return opt

    def _build_sched(self, optimizer, total_steps: int):
        warmup_steps = max(10, int(total_steps * float(getattr(self.cfg.training, 'warmup_ratio', 0.1))))
        cosine_steps = max(1, total_steps - warmup_steps)
        class WarmupThenCosine(optim.lr_scheduler._LRScheduler):
            def __init__(self, opt, warmup, cosine_total, last_epoch=-1):
                self.warmup = warmup
                self.cosine_total = cosine_total
                super().__init__(opt, last_epoch)
            def get_lr(self):
                base_lrs = [g.get('initial_lr', g['lr']) for g in self.optimizer.param_groups]
                step = self.last_epoch + 1
                if step <= self.warmup:
                    scale = step / float(max(1, self.warmup))
                else:
                    t = min(step - self.warmup, self.cosine_total)
                    scale = 0.5 * (1 + math.cos(math.pi * t / float(max(1, self.cosine_total))))
                return [lr * scale for lr in base_lrs]
        for g in optimizer.param_groups:
            if 'initial_lr' not in g:
                g['initial_lr'] = g['lr']
        return WarmupThenCosine(optimizer, warmup_steps, cosine_steps)

    def _compute_loss(self, outputs, cls_logits, batch):
        seq_loss = outputs.loss
        target_labels = batch['cls_label'].to(self.device)
        ce = nn.CrossEntropyLoss()
        cls_loss = ce(cls_logits, target_labels)
        total = self.cfg.training.lm_loss_weight * seq_loss + self.cfg.training.classification_loss_weight * cls_loss
        if self.cfg.training.enable_attention_supervision and hasattr(outputs, 'attentions') and outputs.attentions is not None:
            try:
                attn = outputs.attentions[-1]
                bsz = attn.size(0)
                cls_positions = batch['cls_position']
                idx = torch.arange(bsz, device=attn.device)
                cls_attn = attn[:, :, cls_positions, :]
                cls_pos_tensor = torch.tensor(cls_positions, device=attn.device, dtype=torch.long)
                self_attn_score = cls_attn.mean(dim=1)[idx, 0, cls_pos_tensor]
                attn_loss = (1.0 - self_attn_score).mean()
                total = total + self.cfg.training.attention_loss_weight * attn_loss
            except Exception:
                pass
        return total, seq_loss.detach().item(), cls_loss.detach().item()

    def _validate(self, model: nn.Module, val_loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        per_class = Counter()
        per_class_correct = Counter()
        preds_all = []
        labels_all = []
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
                preds_all.extend(preds.tolist())
                labels_all.extend(labels.tolist())
                for y, p in zip(labels.tolist(), preds.tolist()):
                    per_class[y] += 1
                    if y == p:
                        per_class_correct[y] += 1
        acc = correct / total if total else 0.0
        if total:
            try:
                parts = []
                for c in sorted(per_class.keys()):
                    n = per_class[c]
                    k = per_class_correct.get(c, 0)
                    parts.append(f"{c}:{(k/n):.2f}")
                self.logger.info("Per-class acc: " + ", ".join(parts))
            except Exception:
                pass
        # 保存混淆矩阵
        try:
            if getattr(self.cfg.training, 'save_confusion_matrix', True):
                num_classes = self.cfg.data.num_classes
                cm = [[0 for _ in range(num_classes)] for __ in range(num_classes)]
                for y, p in zip(labels_all, preds_all):
                    cm[y][p] += 1
                out_dir = Path(getattr(self.cfg.training, 'outputs_dir', 'training_v6/outputs'))
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / 'confusion_matrix.json', 'w', encoding='utf-8') as f:
                    json.dump({"matrix": cm}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return acc

    def train_one_epoch(self, model, train_loader, optimizer, scheduler):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
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
            # 梯度裁剪（AMP 下需先反缩放）
            max_norm = float(getattr(self.cfg.training, 'max_grad_norm', 0.0) or 0.0)
            if max_norm > 0.0:
                try:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
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
        latest = self.checkpoint_dir / 'latest_v6.pt'
        torch.save(state, latest)
        if is_best:
            best = self.checkpoint_dir / 'best_v6.pt'
            torch.save(state, best)

    def _try_resume(self, model: nn.Module, optimizer=None, scheduler=None):
        latest = self.checkpoint_dir / 'latest_v6.pt'
        if latest.exists():
            ckpt = torch.load(latest, map_location=self.device)
            model.load_state_dict(ckpt['model'])
            if optimizer is not None and ckpt.get('optimizer') is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
            if scheduler is not None and ckpt.get('scheduler') is not None:
                scheduler.load_state_dict(ckpt['scheduler'])
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
        self.logger.info(f"V6 TRAIN start: epochs={self.cfg.training.num_epochs}, steps/epoch={len(train_loader)}")
        try:
            pid_file = Path('training_v6/logs/v6_training.pid')
            pid_file.write_text(str(os.getpid()))
        except Exception:
            pass
        start_epoch = 0
        if getattr(self.cfg.training, 'resume', True):
            start_epoch = self._try_resume(model, optimizer, scheduler)
        start = time.time()
        for epoch in range(start_epoch, self.cfg.training.num_epochs):
            loss = self.train_one_epoch(model, train_loader, optimizer, scheduler)
            val_acc = self._validate(model, val_loader)
            self.logger.info(f"Epoch {epoch+1}/{self.cfg.training.num_epochs} | loss={loss:.4f} | val_acc={val_acc:.4f}")
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self._save_checkpoint(model, epoch, is_best=True, optimizer=optimizer, scheduler=scheduler)
            else:
                self._save_checkpoint(model, epoch, is_best=False, optimizer=optimizer, scheduler=scheduler)
        dur = time.time() - start
        self.logger.info(f"V6 TRAIN done in {dur/60:.2f} min | best_val_acc={self.best_acc:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['sanity','train'], default='sanity')
    t = V6Trainer()
    args = ap.parse_args()
    if args.mode == 'sanity':
        t.run_sanity()
    else:
        t.train()


if __name__ == '__main__':
    main()
