# research-ideas

## SeeDNorm + GatedNorm Ideas (Single Document)

更新时间：2026-03-05

## 1. SeeDNorm Idea

### 1.1 核心定义

\[
\mathrm{SeeDNorm}(x)=\left[\sigma(x\beta^T)\cdot\alpha+\gamma\right]\odot\frac{x}{\mathrm{RMS}(x)}
\]

### 1.2 直觉

- 继承 RMSNorm 的稳定性（`x / RMS(x)`）。
- 在静态缩放 `gamma` 上叠加输入相关动态项（`sigma(x beta^T) * alpha`）。
- 目标是提升表达灵活性，减少“固定缩放”带来的限制。

### 1.3 价值

1. 参数增量小，适合全模型替换。
2. 对输入分布变化更敏感，理论上更利于泛化。
3. 论文给出多任务改进趋势（语言/视觉）。

### 1.4 风险

1. 动态缩放分支在训练前期可能引入抖动。
2. 对初始化和正则较敏感（尤其 `alpha/beta`）。
3. 工程上需要核融合才能把效率优势吃满。

### 1.5 落地建议

1. 先做单变量替换：`RMSNorm -> SeeDNorm`。
2. `beta` 零初始化，`alpha` 从小值起步。
3. 单独监控门控统计、防止饱和。

## 2. GatedNorm Idea

### 2.1 核心定义

\[
y=\mathrm{RMSNorm}(x),\quad y_g=\sigma\left(W_{up}(\mathrm{swish}(W_{down}(y)))\right),\quad y'=y_g\odot y
\]

### 2.2 直觉

- 在每个 norm 后显式加入动态门控缩放。
- 把“依赖 outlier 做隐式缩放”转成“可学习显式缩放”。

### 2.3 价值

1. 抑制 residual sink / outlier 依赖。
2. 提升激活平滑性，有利量化。
3. 文中报告了性能和量化鲁棒性收益（含 W4A4 场景）。

### 2.4 风险

1. 有额外参数与时延成本。
2. 激活函数和门控粒度选择不当会影响稳定性。
3. 与其他动态缩放机制并用时可能收益递减。

### 2.5 落地建议

1. 先“局部上”：后1/3层或FFN前Norm。
2. 用 elementwise + sigmoid，小 rank 起步（如8/16）。
3. 重点看量化掉点、延迟增量、mid-late loss。

## 3. SeeDNorm + GatedNorm 组合 Idea

## 3.1 总判断

- 组合有潜力，但建议“分阶段增量”，不建议全层直接叠加。

### 3.2 为什么可结合

1. SeeDNorm 提供全局低开销动态缩放底座。
2. GatedNorm 在关键层提供逐维高表达纠偏。
3. 组合后有机会在 outlier 控制和量化鲁棒性上同时获益。

### 3.3 为什么会失败

1. 功能重叠导致收益不线性。
2. 双动态分支导致前期优化共振。
3. 成本叠加超预算。

### 3.4 推荐路线

1. Baseline：RMSNorm。
2. SeeDNorm-only（全替换）。
3. 在 SeeDNorm 基础上加局部 GatedNorm（推荐主线）。
4. 仅在收益显著时再试全层 GatedNorm。

### 3.5 最小实验矩阵

1. RMSNorm baseline
2. SeeDNorm-only
3. GatedNorm-only
4. SeeDNorm + 局部 GatedNorm
5. SeeDNorm + 全层 GatedNorm

统一记录：

1. 训练稳定性（前10% token）
2. sink/outlier 指标与激活尾部统计（P99.9/max）
3. 验证损失与下游均分
4. W8A8/W4A4 掉点
5. 吞吐、延迟、显存、参数增量

### 3.6 成功判据

1. 相比 SeeDNorm-only，loss 稳定下降。
2. W4A4 掉点进一步缩小且跨任务一致。
3. 延迟增量可接受（常见目标 <3~5%）。
4. 无明显训练不稳定。

## 4. 可执行伪代码（推荐主线）

```python
def hybrid_norm(x):
    # SeeDNorm base
    y = x / rms(x)
    s = torch.tanh(y @ beta.T)      # [B,T,1]
    y = y * (gamma + s * alpha)     # broadcast -> [B,T,D]

    # local GatedNorm (selected layers only)
    if use_local_gate:
        g = torch.sigmoid(Wup(swish(Wdown(y))))
        y = y * g
    return y
```

## 5. 参考

- Gated Attention: https://arxiv.org/abs/2505.06708
- GatedNorm / Outlier-driven Rescaling: https://arxiv.org/abs/2601.22966
- SeeDNorm: https://arxiv.org/abs/2510.22777
