# Multi-dimensional Image Segmentation and Recognition using LLMs

本仓库汇总了基于大语言模型（LLM/GPT2 变体）的多维图像序列识别实验工程，包含 v4 / v5 / v6 / v7 四代训练与评测代码，以及可复现实验所需的脚本与可视化占位。

## 亮点与结论
- 模型：`GPT2WithCLSHead`，支持多种分类表示选择策略（`explicit` / `last_non_pad` / `global_pool_mean` / `attn_pool`）。
- 稳定优化（Stable-Opt，使用 `attn_pool` 全局注意聚合）在 v6 验证集取得最佳准确率 **0.925**（第 29/80 轮），在独立测试集（10% 分层切分）取得 **0.900**（100 样本）。
  v7 长训（500 轮，可恢复）进行中：阶段评测（验证/临时测试）当前为 **1.000**（需以独立数据复核泛化）。
- 训练关键：线性 warmup + 余弦退火（无重启）、差分学习率（head lr 放大）、权重衰减与梯度裁剪，显著改善全局信息学习与泛化。

## 版本说明
- `training_v4/`：早期版本，标准 CLS 分类与基础训练流程。
- `training_v5/`：加入多层 pooling、MLP 头与 dropout；修复数据处理中的标签泄漏风险；优化断点恢复。
- `training_v6/`：
  - 模型：支持 `explicit / last_non_pad / global_pool_mean / attn_pool` 四种分类策略；`attn_pool` 使用可学习查询做注意力池化，鼓励学习全局表征。
  - 训练：支持 `head_lr_mult`、`warmup_ratio`、`max_grad_norm`、`scheduler_type`（`warmup_cosine` 或 `cosine_restarts`）、`cosine_restart_T0`。
  - 评测：记录 per-class acc、保存混淆矩阵 JSON 至 `training_v6/outputs/`（本发布版以 `assets/` 存放导出的 JSON）。

## 目录结构
```
training_v4/                 # V4 训练与数据处理（基础版本）
training_v5/                 # V5 训练器+模型（多层池化/MLP/Dropout；数据修复）
training_v6/                 # V6 训练器+模型（多策略分类；稳定优化支持）
assets/                      # 评测资产（日志、confusion_matrix.json 等）
docs/                        # 可视化图片（训练曲线、混淆矩阵）
scripts/                     # 工具脚本（可选地生成图像）
requirements.txt             # 运行依赖
.gitignore                   # 忽略大文件/缓存
```

## 复现步骤（V6 稳定优化推荐）
1) 安装依赖
```bash
pip install -r requirements.txt
```

2) 运行训练（80 轮稳定优化配置）
```bash
python - <<'PY'
from training_v6.core.trainer_v6 import V6Trainer
trainer = V6Trainer()
trainer.cfg.model.cls_strategy = 'attn_pool'
trainer.cfg.training.num_epochs = 80
trainer.cfg.training.learning_rate = 2e-4
trainer.cfg.training.head_lr_mult = 5.0
trainer.cfg.training.weight_decay = 0.05
trainer.cfg.training.warmup_ratio = 0.2
trainer.cfg.training.max_grad_norm = 1.0
trainer.cfg.training.scheduler_type = 'warmup_cosine'
trainer.train()
PY
```

3) 测试评估（基于最佳权重 `best_v6.pt`，10% 分层测试集）
```bash
python - <<'PY'
import json, torch
from pathlib import Path
from torch.utils.data import DataLoader
from training_v6.core.trainer_v6 import V6Trainer
from training_v6.core.data_processor_v6 import V6DataProcessor, V6Dataset

trainer = V6Trainer()
vocab = trainer._load_vocab()
proc = V6DataProcessor(vocab, max_length=trainer.cfg.data.max_length, num_classes=trainer.cfg.data.num_classes)
rows = proc.load_and_process(trainer.cfg.data.train_data_path)
train_rows, val_rows = V6DataProcessor.stratified_split(rows, seed=trainer.cfg.data.seed, train_ratio=0.8)
val_half, test_rows = V6DataProcessor.stratified_split(val_rows, seed=trainer.cfg.data.seed+1, train_ratio=0.5)
test_loader = DataLoader(V6Dataset(test_rows), batch_size=trainer.cfg.training.batch_size, shuffle=False)

model = trainer._build_model()
ckpt = torch.load('training_v6/outputs/checkpoints/best_v6.pt', map_location=trainer.device)
model.load_state_dict(ckpt['model'])
model.eval()

import torch.nn.functional as F
correct = total = 0
with torch.no_grad():
    for batch in test_loader:
        ids = batch['input_ids'].to(trainer.device)
        mask = batch['attention_mask'].to(trainer.device)
        cls_pos = batch['cls_position']
        y = batch['cls_label'].to(trainer.device)
        _, logits = model(input_ids=ids, attention_mask=mask, labels=batch['labels'].to(trainer.device), cls_positions=cls_pos)
        pred = torch.argmax(logits, dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
print('TEST_ACC=', correct/total if total else 0.0)
PY
```

## 模型结构与分类策略（V6）
- `explicit`：使用明确的 `<CLS>` 位置隐状态做分类。
- `last_non_pad`：使用序列最后一个非 PAD/EOS 位置隐状态（对末尾强信号敏感）。
- `global_pool_mean`：对非 PAD 位置做均值池化（鼓励全局信息，但信号被均摊）。
- `attn_pool`（推荐）：使用可学习查询对全序列隐状态做注意力聚合，兼顾全局感受野与可分性。

## 训练要点与坑点
- 混合精度：`torch.cuda.amp.GradScaler` 与 `autocast`（注意 PyTorch 新版 API 的 deprecate 提示）。
- 学习率：主干 `2e-4`，头部 `lr * head_lr_mult = 1e-3`；权重衰减 `0.05`；warmup 比例 `0.2`。
- 调度：`warmup_cosine`（无重启）更稳定；如需 `cosine_restarts` 可配置 `cosine_restart_T0`。
- 梯度：`max_grad_norm=1.0` 抑制梯度爆炸，提升收敛稳定性。
- 数据处理：确保 `<CLS>` 不被替换为编码标签的 `<CLS_i>`，避免“标签泄漏”。

## 指标与可视化
- 验证集最佳：`0.925`（第 29/80 轮）。
- 测试集（10% 分层）：`0.900`（100 样本）。
- 可视化：
  - 训练曲线：`docs/v6_val_acc_curve.png`
  - 混淆矩阵（验证）：`docs/v6_confusion_matrix.png`
  - 混淆矩阵（测试）：`docs/v6_test_confusion_matrix.png`

> 若图片暂缺，可用 `scripts/gen_visuals.py` 读取 `assets/` 下日志与 JSON 自动生成。

## 数据路径与配置
- 在 `training_v6/config/v6_config.py` 中设置：
  - `DataConfig.train_data_path`
  - `DataConfig.vocab_path`
  - 其余训练超参位于 `TrainingConfig`。

## 仓库
- GitHub: [`Ludan-daye/Multi-dimensional-image-segmentation-and-recognition-using-large-language-models`](https://github.com/Ludan-daye/Multi-dimensional-image-segmentation-and-recognition-using-large-language-models)

