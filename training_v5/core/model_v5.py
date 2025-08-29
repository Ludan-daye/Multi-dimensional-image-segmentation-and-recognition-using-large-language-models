#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config


class MLPHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class GPT2WithCLSHead(nn.Module):
    def __init__(self, config: GPT2Config, num_classes: int = 10, head_config=None):
        super().__init__()
        # 确保输出隐藏态
        config.output_hidden_states = True
        self.backbone = GPT2LMHeadModel(config)
        hidden = config.n_embd

        # 头部配置
        self.use_multi_layer_pool = getattr(head_config, 'use_multi_layer_pool', False)
        self.pool_last_k = int(getattr(head_config, 'pool_last_k', 1))
        self.use_mlp_head = getattr(head_config, 'use_mlp_head', False)
        self.head_dropout = float(getattr(head_config, 'head_dropout', 0.0))
        mlp_hidden = int(getattr(head_config, 'mlp_hidden', hidden))

        if self.use_mlp_head:
            self.cls_head = MLPHead(hidden, mlp_hidden, num_classes, self.head_dropout)
        else:
            self.cls_head = nn.Sequential(
                nn.Dropout(self.head_dropout),
                nn.Linear(hidden, num_classes)
            )

    def forward(self, input_ids, attention_mask=None, labels=None, cls_positions=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        cls_logits = None
        if cls_positions is not None and outputs.hidden_states is not None:
            # hidden_states: tuple(len = n_layer+1 (embeddings)), each [B, T, H]
            if self.use_multi_layer_pool:
                k = max(1, self.pool_last_k)
                selected = outputs.hidden_states[-k:]  # list of [B, T, H]
                stacked = torch.stack(selected, dim=0)  # [K, B, T, H]
                pooled_last = stacked.mean(dim=0)       # [B, T, H]
            else:
                pooled_last = outputs.hidden_states[-1]

            bsz = input_ids.size(0)
            cls_pos_tensor = torch.as_tensor(cls_positions, device=pooled_last.device, dtype=torch.long)
            idx = torch.arange(bsz, device=pooled_last.device)
            cls_hidden = pooled_last[idx, cls_pos_tensor, :]  # [B, H]
            cls_logits = self.cls_head(cls_hidden)  # [B, num_classes]
        return outputs, cls_logits

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_head_params(self):
        return self.cls_head.parameters()


