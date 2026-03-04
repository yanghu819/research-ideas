# Idea: SeeDNorm + GatedNorm Integration

## 一句话判断

这个组合有潜力，但应采用“分阶段增量”而不是全层直接叠加。

## 为什么可结合

- SeeDNorm 给全模型提供低开销、输入相关动态缩放底座。
- GatedNorm 给关键层提供更强逐维门控能力。
- 二者都属于显式重缩放机制，理论上可协同降低 outlier 依赖。

## 为什么不能粗暴全叠

1. 功能重叠：都在做动态scale，收益可能递减。
2. 优化冲突：双动态分支可能让训练初期振荡更明显。
3. 成本叠加：额外参数/延迟叠加。

## 推荐路线（从稳到激进）

1. Baseline + SeeDNorm（全替换RMSNorm）
2. Baseline + SeeDNorm + Gated Attention（若已有GA框架）
3. 在(2)基础上仅对后1/3层加入轻量GatedNorm（rank 8/16）
4. 最后才评估全层GatedNorm

## 必做实验矩阵

1. RMSNorm baseline
2. SeeDNorm only
3. GatedNorm only
4. SeeDNorm + 局部GatedNorm
5. SeeDNorm + 全层GatedNorm

## 必看指标

1. 训练稳定性（前10% token 的loss曲线）
2. residual/attention sink 指标与激活尾部统计（P99.9/max）
3. 下游平均分
4. W8A8 / W4A4 掉点
5. 吞吐、延迟、显存、参数增量

## 工程建议

1. 两个动态分支都使用有界激活（sigmoid/tanh）。
2. SeeDNorm 的 \(\beta\) 零初始化，\(\alpha\) 从小值起步。
3. 给 GatedNorm 分支加 warmup 或渐进启用。
4. 动态参数单独正则，避免门控饱和。

## 结论

最推荐的首发版本是：`SeeDNorm 全替换 + 局部 GatedNorm`。

## 参考

- https://arxiv.org/abs/2505.06708
- https://arxiv.org/abs/2601.22966
- https://arxiv.org/abs/2510.22777
