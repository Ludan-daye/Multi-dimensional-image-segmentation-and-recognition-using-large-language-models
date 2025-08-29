from transformers import GPT2LMHeadModel, GPT2Config
import torch
import torch.nn as nn
import math


class GPT2WithCLSHead(nn.Module):
    def __init__(self, config: GPT2Config, num_classes: int = 10, head_config=None):
        super().__init__()
        config.output_hidden_states = True
        if hasattr(config, 'output_attentions'):
            config.output_attentions = True
        self.backbone = GPT2LMHeadModel(config)
        hidden = config.n_embd
        self.cls_head = nn.Linear(hidden, num_classes)
        # 策略配置
        self.cls_strategy = getattr(head_config, 'cls_strategy', 'explicit') if head_config is not None else 'explicit'
        self.pad_token_id = getattr(head_config, 'pad_token_id', 0)
        self.eos_token_id = getattr(head_config, 'eos_token_id', 5)
        # 注意力池化参数（用于 cls_strategy == 'attn_pool'）
        self.attn_query = nn.Parameter(torch.randn(hidden))
        self.attn_dropout = nn.Dropout(getattr(head_config, 'dropout', 0.1) if head_config is not None else 0.1)

    def _select_repr(self, hidden_last: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, cls_positions):
        # hidden_last: [B, T, H]
        bsz, seq_len, hidden = hidden_last.size()
        device = hidden_last.device
        if self.cls_strategy == 'explicit' and cls_positions is not None:
            idx_b = torch.arange(bsz, device=device)
            pos = torch.as_tensor(cls_positions, device=device, dtype=torch.long)
            return hidden_last[idx_b, pos, :]
        if self.cls_strategy == 'last_non_pad':
            # 选取最后一个非PAD且优先EOS位置
            is_non_pad = (input_ids != self.pad_token_id).to(hidden_last.dtype)
            # 默认最后非pad索引
            lengths = is_non_pad.argmax(dim=1)  # 错误方向，需从右向左
            # 从右向左扫描：将mask反转
            rev = torch.flip(is_non_pad, dims=[1])
            last_non_pad_from_right = rev.argmax(dim=1)
            pos = (seq_len - 1) - last_non_pad_from_right
            # 若存在EOS，优先EOS
            has_eos = (input_ids == self.eos_token_id)
            eos_pos = torch.where(has_eos.any(dim=1), has_eos.float().argmax(dim=1), pos)
            final_pos = eos_pos
            idx_b = torch.arange(bsz, device=device)
            return hidden_last[idx_b, final_pos, :]
        if self.cls_strategy == 'global_pool_mean':
            # 按注意力mask计算均值池化
            mask = attention_mask.to(hidden_last.dtype).unsqueeze(-1)  # [B, T, 1]
            summed = (hidden_last * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return summed / denom
        if self.cls_strategy == 'attn_pool':
            # 可学习查询对非PAD位置做注意力汇聚
            # scores: [B, T]
            query = self.attn_query
            scores = torch.matmul(hidden_last, query) / math.sqrt(hidden_last.size(-1))
            # mask PAD 为 -inf
            if attention_mask is None:
                attention_mask = (input_ids != self.pad_token_id).long()
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            attn_weights = torch.softmax(scores, dim=1)
            attn_weights = self.attn_dropout(attn_weights)
            # 加权求和得到全局表示
            rep = torch.bmm(attn_weights.unsqueeze(1), hidden_last).squeeze(1)  # [B, H]
            return rep
        # 回退
        idx_b = torch.arange(bsz, device=device)
        pos = torch.as_tensor(cls_positions if cls_positions is not None else [0]*bsz, device=device, dtype=torch.long)
        return hidden_last[idx_b, pos, :]

    def forward(self, input_ids, attention_mask=None, labels=None, cls_positions=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states is not None else None
        cls_logits = None
        if hidden_states is not None:
            rep = self._select_repr(hidden_states, input_ids, attention_mask if attention_mask is not None else (input_ids != self.pad_token_id).long(), cls_positions)
            cls_logits = self.cls_head(rep)
        return outputs, cls_logits
