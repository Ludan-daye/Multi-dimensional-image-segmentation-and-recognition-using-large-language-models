# Training V4

本目录包含V4训练系统（代码、配置、脚本）、训练检查点与评估结果。

## 亮点
- 分层划分（seed=42）：训练/验证均覆盖10类，避免顺序切分偏差
- 明确分类目标：在`<CLS>`位置预测`<CLS_0>...<CLS_9>`
- 数据处理：左截断保尾，保证`<CLS>`与`<EOS>`不被截断，长度统一1024
- 混合精度与GPU并行，断点可恢复（检查点保存在`outputs/checkpoints/`）

## 目录结构
- `config/`：自适应GPU与全局参数
- `core/`：`enhanced_data_processor.py`、`training_objectives.py`、`v4_trainer.py`
- `scripts/`：启动与监控脚本
- `outputs/checkpoints/`：`latest_checkpoint.pt`（Git LFS）
- `results/VAL_RESULTS_V4.json`：验证集整体与分类别指标

## 关键结果（分层切分）
- Overall 验证准确率：0.2800（56/200）
- 每类准确率：
  - 0: 0.5000, 1: 0.4000, 2: 0.4000, 3: 0.5000,
  - 4: 0.3000, 5: 0.3000, 6: 0.2000, 7: 0.2000,
  - 8: 0.0000, 9: 0.0000

> 注：之前验证=0.0000由顺序切分导致验证集仅含8/9类。现已修复为分层切分。

## 运行
```bash
# 进入目录
cd training_v4

# 后台训练（可断点恢复）
nohup python3 core/v4_trainer.py > logs/v4_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 离线评估（使用latest checkpoint）
python3 - <<'PY'
import sys, torch
sys.path.append('.')
from core.v4_trainer import V4Trainer
T=V4Trainer(); T._initialize_model(); T._load_and_prepare_data(); T._load_checkpoint();
acc=T._validate(); print('VAL_ACC=', acc)
PY
```

## 依赖
- 词表默认从`training_v3`路径自动回退查找（建议保留`training_v3/outputs/best_model_* /vocab.json`）

## 版本
- Epochs=500，长度=1024；分类权重0.8、序列权重0.2

