# V5 训练总结

- 时间: 2025-08-27 08:57:31
- 日志: training_v5/logs/v5_training_20250826_230443.log
- 最佳验证准确率: 12.50%
- 训练轮次: 500/500
- 最新若干轮 val_acc: 0.0850（基本稳定）

## 关键信息
- 最佳检查点: training_v5/outputs/checkpoints/best_v5.pt
- 最新检查点: training_v5/outputs/checkpoints/latest_v5.pt

## 分析与结论
- 验证准确率长期停在 0.085 附近，最佳为 0.1250。
- 可能原因：
  1) LM 损失未屏蔽 PAD，噪声梯度压制分类学习；
  2) 训练目标从 V4 的“CLS 位置 LM 预测 <CLS_X>”改为“独立分类头+CE”，学习更难；
  3) 调度无 warmup，学习率与步幅可能不匹配；
  4) 缺少样本一致性校验；
  5) WeightedRandomSampler 可能放大噪声。

## 改进建议（优先级）
1. mask PAD：数据管道将 padding 的 labels 置为 -100，仅对非 PAD 位置计算 LM 损失。
2. 添加样本一致性校验：对齐 V4，校验 cls_label、<CLS_X> 与 cls_position。
3. 训练目标对齐 V4：在 CLS 位置用 LM logits 预测 <CLS_X>（可与分类头并行加权）。
4. 学习率与调度：增加 warmup（LinearWarmup+Cosine 或 OneCycleLR），先将 lr 调为 5e-5 做快速对比。
5. 采样策略：先关闭 WeightedRandomSampler，用 shuffle 验证收敛。
6. 进行 20–30 轮短训 A/B 验证上述改动的增益。

