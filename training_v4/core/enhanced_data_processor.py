#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training V4 增强数据处理器
按照用户要求：1024全局长度，确保CLS token不被截断
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

class EnhancedDataProcessor:
    """
    增强数据处理器 - V4专用
    确保训练数据符合所有要求
    """

    def __init__(self, vocab: Dict[str, int], max_length: int = 1024):
        self.vocab = vocab
        self.max_length = max_length
        # 特殊token IDs
        self.pad_token_id = vocab.get('<PAD>', 0)
        self.cls_token_id = vocab.get('<CLS>', 4)
        self.eos_token_id = vocab.get('<EOS>', 5)
        # 分类token映射
        self.cls_tokens = {i: vocab.get(f'<CLS_{i}>', 509 + i) for i in range(10)}
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"📊 数据处理器初始化: max_length={max_length}")
        self.logger.info(f"🎯 CLS token映射: {self.cls_tokens}")

    def process_sequence(self, token_ids: List[int], label: int) -> Optional[Dict[str, Any]]:
        """
        处理单个序列 - 确保CLS token完整
        """
        # 验证标签
        if not (0 <= label <= 9):
            self.logger.warning(f"标签{label}超出范围[0,9]，跳过")
            return None

        # 左截断策略：保护尾部CLS token
        if len(token_ids) > self.max_length:
            # 保留最后max_length个token，保护CLS和EOS
            token_ids = token_ids[-self.max_length:]
            self.logger.debug(f"序列截断: 原长度 -> {self.max_length}")

        # 填充到固定长度
        if len(token_ids) < self.max_length:
            padding_length = self.max_length - len(token_ids)
            token_ids = token_ids + [self.pad_token_id] * padding_length

        # 查找CLS token位置
        cls_position = -1
        for i, token_id in enumerate(token_ids):
            if token_id == self.cls_token_id:
                cls_position = i
                break

        if cls_position == -1:
            self.logger.warning("未找到<CLS> token，跳过此样本")
            return None

        # 获取目标CLS_X token
        cls_target_token = self.cls_tokens.get(label)
        if cls_target_token is None:
            self.logger.error(f"找不到标签{label}对应的CLS token")
            return None

        # 创建attention mask
        attention_mask = [1 if t != self.pad_token_id else 0 for t in token_ids]

        # 验证数据完整性
        if not self._validate_sample(token_ids, cls_position, label, cls_target_token):
            return None

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(token_ids, dtype=torch.long),  # 序列建模目标
            'cls_position': cls_position,
            'cls_label': label,
            'cls_target_token': cls_target_token
        }

    def _validate_sample(self, token_ids: List[int], cls_position: int, label: int, cls_target: int) -> bool:
        """验证样本完整性"""
        try:
            # 检查序列长度
            if len(token_ids) != self.max_length:
                self.logger.error(f"序列长度错误: {len(token_ids)} != {self.max_length}")
                return False

            # 检查CLS位置
            if cls_position < 0 or cls_position >= len(token_ids):
                self.logger.error(f"CLS位置无效: {cls_position}")
                return False

            # 验证CLS token
            if token_ids[cls_position] != self.cls_token_id:
                self.logger.error(f"CLS位置token错误: {token_ids[cls_position]} != {self.cls_token_id}")
                return False

            # 检查标签和目标token一致性
            expected_target = self.cls_tokens.get(label)
            if cls_target != expected_target:
                self.logger.error(f"目标token不一致: {cls_target} != {expected_target}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"样本验证异常: {e}")
            return False

    def load_and_process_data(self, data_file: str) -> List[Dict[str, Any]]:
        """
        加载并处理数据文件
        """
        self.logger.info(f"📁 加载数据文件: {data_file}")

        if not Path(data_file).exists():
            raise FileNotFoundError(f"数据文件不存在: {data_file}")

        processed_data: List[Dict[str, Any]] = []
        total_lines = 0
        valid_samples = 0
        cls_found_count = 0

        with open(data_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                total_lines += 1

                try:
                    item = json.loads(line.strip())

                    # 处理不同的数据格式
                    if 'input_ids' in item:
                        token_ids = item['input_ids']
                    elif 'tokens' in item:
                        # 处理tokens字符串格式
                        tokens_str = item['tokens']
                        if isinstance(tokens_str, str):
                            token_names = tokens_str.split()
                            token_ids = [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in token_names]
                        else:
                            token_ids = tokens_str
                    else:
                        continue

                    label = item.get('label', -1)

                    # 处理序列
                    processed_item = self.process_sequence(token_ids, label)

                    if processed_item is not None:
                        processed_data.append(processed_item)
                        valid_samples += 1
                        cls_found_count += 1

                    # 进度报告
                    if (line_idx + 1) % 100 == 0:
                        self.logger.info(f"处理进度: {line_idx + 1}/{total_lines} 行，有效: {valid_samples}")

                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON解析错误 第{line_idx + 1}行: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"处理错误 第{line_idx + 1}行: {e}")
                    continue

        # 数据统计
        self.logger.info(f"📊 数据处理完成:")
        self.logger.info(f"   - 总行数: {total_lines}")
        self.logger.info(f"   - 有效样本: {valid_samples}")
        if total_lines:
            self.logger.info(f"   - 成功率: {valid_samples/total_lines*100:.2f}%")
            self.logger.info(f"   - CLS找到率: {cls_found_count/total_lines*100:.2f}%")

        if valid_samples == 0:
            raise ValueError("没有有效的训练样本")

        return processed_data

class ImageClassificationDataset(Dataset):
    """图像分类数据集 - V4专用"""
    
    def __init__(self, processed_data: List[Dict[str, Any]]):
        self.data = processed_data
        print(f"📊 数据集初始化: {len(self.data)} 样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
