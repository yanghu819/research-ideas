# Idea: SeeDNorm

## 核心定义

\[
\mathrm{SeeDNorm}(x)=\left[\sigma(x\beta^T)\cdot\alpha+\gamma\right]\odot\frac{x}{\mathrm{RMS}(x)}
\]

- 在 RMSNorm 的基础上，把静态缩放 \(\gamma\) 扩展为“静态 + 输入相关动态缩放”。
- 动态分支依赖当前输入，强调保留输入范数相关信息。

## 主要价值

1. 保持 RMSNorm 的训练稳定范式，同时提高前向表达灵活性。
2. 参数开销小（动态分支是低秩/低开销形式）。
3. 论文在语言和视觉任务上报告了收敛与性能改进趋势。

## 风险点

1. 动态系数在训练早期可能带来额外不稳定性。
2. 初始化与正则化敏感（尤其 \(\alpha,\beta\)）。
3. 需要核函数融合来保证极致效率。

## 落地建议

1. 先做全量 `RMSNorm -> SeeDNorm` 单变量替换。
2. 保持现有训练配方不变，仅调 \(\alpha\) 初始化和 weight decay。
3. 优先观察前10%训练阶段稳定性与验证损失斜率。

## 参考

- https://arxiv.org/abs/2510.22777
