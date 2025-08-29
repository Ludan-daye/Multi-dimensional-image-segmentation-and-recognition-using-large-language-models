# TRAINING V4 报告

## 概览
- 模型: GPT2LMHead（768d, 12层, n_positions=1024）
- 训练轮数: 500 epochs（已完成）
- 主要目标: 在`<CLS>`位置预测正确的`<CLS_X>`分类token
- 数据长度策略: 左截断保尾，padding到1024，保护`<CLS>/<EOS>`

## 配置与实现
- 配置管理: `config/advanced_config.py`
- 数据处理: `core/enhanced_data_processor.py`
- 训练器: `core/v4_trainer.py`
- 训练目标: `core/training_objectives.py`（分类损失0.8 + 序列损失0.2）
- 启动/监控: `scripts/start_v4_training.sh`, `scripts/monitor_v4_training.sh`
- 断点恢复: `outputs/checkpoints/latest_checkpoint.pt`（Git LFS）

## 数据与切分
- 源数据: `training_v3/generated_sequences_super_enhanced/sequences_labels_fixed_v2_maxlen1024.jsonl`
- 词表: 自动回退到`training_v3/outputs/best_model_silent/vocab.json`
- 切分策略: 分层切分（seed=42），每类80%训练/20%验证，保证验证覆盖10类

## 指标
### 可视化
![Confusion Matrix](results/v4_confusion_matrix.png)

![Top-k Accuracy](results/v4_topk.png)

- Overall 验证准确率: 0.2800（56/200）
- 每类准确率:
  - 0: 0.5000 (10/20)
  - 1: 0.4000 (8/20)
  - 2: 0.4000 (8/20)
  - 3: 0.5000 (10/20)
  - 4: 0.3000 (6/20)
  - 5: 0.3000 (6/20)
  - 6: 0.2000 (4/20)
  - 7: 0.2000 (4/20)
  - 8: 0.0000 (0/20)
  - 9: 0.0000 (0/20)
- 备注: 早先的验证=0.0000是因顺序切分导致验证集只含(8,9)，现已修复。

## 可复现实验
- 离线评估（使用最新checkpoint）:
```bash
python3 - <<'PY'
import sys, torch
sys.path.append('.')
from core.v4_trainer import V4Trainer
T=V4Trainer(); T._initialize_model(); T._load_and_prepare_data(); T._load_checkpoint();
print('VAL_ACC=', T._validate())
PY
```
- 训练（后台可恢复）:
```bash
nohup python3 core/v4_trainer.py > logs/v4_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 诊断与改进建议
1) 类别不平衡/覆盖不足：8/9类性能为0. 建议：
   - 使用`WeightedRandomSampler`或`ClassBalancedSampler`进行再训练/微调
   - 构造少量针对8/9类的强化样本，开展小学习率微调
2) 优化超参：
   - 降低学习率（例如 5e-5），增加warmup，观察验证曲线
   - 调整分类/序列损失权重（如 0.9/0.1）
3) 监控：
   - 增加按类准确率与混淆矩阵的定期输出
4) 数据：
   - 样本内`<CLS>`位置目前多为~786，可考虑轻微抖动，避免位置过拟合

## 版本
- 提交包含：分层切分代码、评估结果JSON、README与本报告

