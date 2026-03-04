# SeeDNorm + GatedNorm 组合方案深度分析

## 1. 核心判断

- 可结合，但应采用“先替换、后增强”的策略。
- 组合的价值不是简单叠加，而是“低开销全局动态缩放 + 局部高表达门控修正”。

## 2. 为什么组合有机会

1. SeeDNorm 在全模型提供输入相关缩放底座。
2. GatedNorm 在关键层补充逐维高表达控制。
3. 若 outlier 主要出现在部分层，局部 GatedNorm 的性价比更高。

## 3. 为什么会失败

1. 功能重叠导致收益不线性。
2. 双动态scale分支造成优化共振（早期不稳）。
3. 成本叠加超过可部署预算。

## 4. 推荐的分阶段路线

### 阶段1：全量 SeeDNorm

- `RMSNorm -> SeeDNorm` 全替换。
- 保持其余结构不变，建立稳定新基线。

### 阶段2：局部 GatedNorm

- 仅在后1/3层或 FFN 前 norm 增加 GatedNorm。
- rank 从 8/16 起步，激活用 sigmoid。

### 阶段3：可选扩展

- 若阶段2收益显著且成本可接受，再尝试扩大覆盖层数。

## 5. 最小可执行实验矩阵

1. RMSNorm baseline
2. SeeDNorm-only
3. GatedNorm-only
4. SeeDNorm + 局部 GatedNorm
5. SeeDNorm + 全层 GatedNorm

统一评估：

1. 训练稳定性：前10% tokens 的 loss 曲线与梯度异常比例
2. outlier 指标：
- attention sink 比例
- residual sink 峰值
- 激活分布尾部（P99.9 / max）
3. 能力指标：验证损失 + 下游任务均分
4. 量化指标：W8A8/W4A4 掉点
5. 工程指标：吞吐、延迟、显存、参数增量

## 6. 成功标准（建议）

1. 相比 SeeDNorm-only，验证损失有稳定改进。
2. W4A4 掉点进一步缩小，且跨任务一致。
3. 延迟增量在可接受范围（常见目标 <3~5%）。
4. 无显著训练不稳定现象。

## 7. 实现层面的注意事项

1. 两个动态分支都使用有界激活（sigmoid/tanh）。
2. SeeDNorm 的 \(\beta\) 用零初始化，\(\alpha\) 小值起步。
3. GatedNorm 分支建议 warmup 渐进启用。
4. 对动态参数单独正则，避免门控饱和。

## 8. 可执行伪代码

```python
def hybrid_norm(x):
    # SeeDNorm base
    y = x / rms(x)
    s = torch.tanh(y @ beta.T)      # [B,T,1]
    y = y * (gamma + s * alpha)     # broadcast to [B,T,D]

    # optional local GatedNorm
    if use_local_gate:
        g = torch.sigmoid(Wup(swish(Wdown(y))))
        y = y * g
    return y
```

## 9. 最终建议

- 首发版本优先：`SeeDNorm 全替换 + 局部 GatedNorm`。
- 不建议第一版直接全层双动态分支。
- 先拿到“量化收益 + 成本可控 + 稳定训练”三者同时满足，再扩展覆盖。

## 参考

- https://arxiv.org/abs/2505.06708
- https://arxiv.org/abs/2601.22966
- https://arxiv.org/abs/2510.22777
