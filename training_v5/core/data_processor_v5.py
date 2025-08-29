#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset
import random


class V5DataProcessor:
    def __init__(self, vocab: Dict[str, int], max_length: int = 1024, num_classes: int = 10):
        self.vocab = vocab
        self.max_length = max_length
        self.num_classes = num_classes
        self.pad_id = vocab.get('<PAD>', 0)
        self.cls_id = vocab.get('<CLS>', 4)
        self.eos_id = vocab.get('<EOS>', 5)
        self.cls_tokens = {i: vocab.get(f'<CLS_{i}>', 506 + i) for i in range(num_classes)}
        self.logger = logging.getLogger(__name__)

    def process_sequence(self, token_ids: List[int], label: int) -> Dict[str, Any]:
        if len(token_ids) > self.max_length:
            token_ids = token_ids[-self.max_length:]
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [self.pad_id] * (self.max_length - len(token_ids))

        try:
            cls_pos = token_ids.index(self.cls_id)
        except ValueError:
            raise ValueError('CLS not found')

        target_token = self.cls_tokens[label]
        attn = [0 if t == self.pad_id else 1 for t in token_ids]
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn, dtype=torch.long),
            # 将 padding 的 labels 置为 -100，避免在 LM 损失中计算 PAD
            'labels': torch.tensor([
                (-100 if t == self.pad_id else t) for t in token_ids
            ], dtype=torch.long),
            'cls_position': cls_pos,
            'cls_label': label,
            'cls_target_token': target_token,
        }

    def load_and_process(self, path: str) -> List[Dict[str, Any]]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        data: List[Dict[str, Any]] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if 'input_ids' in item:
                    token_ids = item['input_ids']
                else:
                    tokens = item.get('tokens')
                    if isinstance(tokens, str):
                        names = tokens.split()
                        token_ids = [self.vocab.get(t, self.vocab.get('<UNK>', 1)) for t in names]
                    else:
                        token_ids = tokens
                label = int(item.get('label', 0))
                try:
                    data.append(self.process_sequence(token_ids, label))
                except Exception:
                    continue
        if not data:
            raise ValueError('no valid samples')
        return data

    @staticmethod
    def stratified_split(data: List[Dict[str, Any]], seed: int = 42, train_ratio: float = 0.8):
        rnd = random.Random(seed)
        buckets: Dict[int, List[Dict[str, Any]]] = {}
        for x in data:
            buckets.setdefault(int(x['cls_label']), []).append(x)
        train, val = [], []
        for k, items in buckets.items():
            rnd.shuffle(items)
            n = len(items)
            n_train = max(1, min(n-1, int(n * train_ratio))) if n > 1 else 1
            train.extend(items[:n_train])
            val.extend(items[n_train:])
        rnd.shuffle(train)
        rnd.shuffle(val)
        return train, val


class V5Dataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        return self.rows[idx]


