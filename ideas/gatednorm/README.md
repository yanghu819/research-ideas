# Idea: GatedNorm

## 核心定义

\[
y=\mathrm{RMSNorm}(x),\quad y_g = \sigma(W_{up}(\mathrm{swish}(W_{down}(y)))),\quad y' = y_g \odot y
\]

- 在每个归一化层后加低秩、逐维门控。
- 目标是显式提供重缩放能力，减少模型对 outlier/sink 的隐式依赖。

## 主要价值

1. 与 outlier-driven rescaling 视角一致：把“靠异常值缩放”改成“显式门控缩放”。
2. 抑制 residual sink，改善激活平滑性。
3. 论文报告了性能收益和量化鲁棒性提升（含 W4A4 场景）。

## 风险点

1. 额外参数与训练/推理开销。
2. 激活函数与门控粒度敏感（有界激活更稳）。
3. 全层启用时可能和已有动态机制功能重叠。

## 落地建议

1. 先局部加（后1/3层或FFN前Norm）再扩展。
2. 优先 elementwise + sigmoid，小 rank 起步。
3. 重点看量化掉点、延迟增量和中后期 loss 差距。

## 参考

- https://arxiv.org/abs/2601.22966
