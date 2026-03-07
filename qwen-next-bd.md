# Qwen-Next + Block Diffusion（近似版思路）

更新时间：2026-03-07

## 1. 目标

目标很死，没有绕路：

1. 必须用 `Qwen-Next` 这类 hybrid 架构。
2. 必须保留生成时的 `Block Diffusion` 行为。
3. 接受工程上做近似，但不能把方法语义搞烂。

这里的核心问题不是 attention，而是 `SSM / FLA / DeltaNet` 这半套。

## 2. 为什么严格版很难

经典 `Block Diffusion` 在纯 attention 模型里，训练通常可以写成：

- 输入：`[x_t | x_0]`
- 机制：`custom BD mask`
- 目标：只在 noisy 半边恢复当前 block 的 token

这套在 pure attention 上好使，因为 attention 天生有：

1. `KV cache`，每个历史 token 都有显式缓存。
2. 任意二维可见性控制，mask 怎么写都行。
3. 改一个 token 后，局部重算语义比较自然。

但 `Qwen-Next` 不是纯 attention。它是 hybrid：

- 一部分层/子层是 attention
- 一部分层/子层是 `FLA / DeltaNet / SSM`

问题在于 `SSM / FLA` 的 cache 不是 `KV cache`，而是压缩后的递归状态：

- `recurrent_state`
- `conv_state`

它不是“每个 token 对应一个 state 表”，而是“前缀累计后的压缩状态”。

这直接带来三个麻烦：

1. 不能像 attention 一样随机访问历史 token 状态。
2. 当前 block 里改了前面的 token，后面的 state 理论上都变了。
3. 很难把 `SSM` 强行塞进 `[x_t | x_0] + 二维 BD mask` 这个语义里。

所以，**严格复刻 pure-attention 的 BD 训练形式，在 Qwen-Next 上并不自然。**

## 3. 关键判断

### 3.1 不能假装严格一致

如果坚持 `Qwen-Next + BD`，那就得承认：

**我们做不了 token 级严格一致的 BD，只能做 block 级一致的近似 BD。**

这不是投降，这是结构约束。

### 3.2 可以守住的底线

虽然做不到 token 级 exactness，但下面这三条必须守住：

1. `history` 和 `draft` 必须隔离。
2. 历史状态只能在当前 block `finalize` 之后提交。
3. 当前 block 每次 refine 都必须从同一个 block-boundary snapshot 重跑。

只要这三条守住了，方法虽然不是“原教旨 exact BD”，但至少是自洽的 hybrid-BD。

## 4. 我们的近似版本：核心 idea

核心思路是：

**别让 SSM 学二维 mask。让它做自己擅长的事情：前缀递推。**

于是把原来 single-forward 的 BD，改写成 multi-forward 的 block-prefix 训练和推理。

### 4.1 训练时的语义

对于第 `b` 个 block：

- 历史 block 用 clean token：`x_0(<b)`
- 当前 block 用 noisy token：`x_t(b)`
- 目标：恢复当前 block 的 clean token `x_0(b)`

也就是：

- `block1`：输入 `x_t(1)`
- `block2`：输入 `x_0(1) + x_t(2)`
- `block3`：输入 `x_0(1) + x_0(2) + x_t(3)`
- 一般地：输入 `x_0(<b) + x_t(b)`

这和 dFactory 里 `[x_t | x_0] + BD mask` 的目标本质一致：

- 历史是 clean
- 当前块是 noisy
- 只恢复当前块

区别只是：

- pure attention 版本把这个语义塞进一次 forward
- 我们把它拆成多次 prefix forward

### 4.2 推理时的语义

推理也用同一套近似：

1. 已经完成的 block 组成 clean history。
2. 当前 block 初始化为全 `mask` 或部分 draft，记作 `x_t(b)`。
3. 每一轮 refine：
   - 输入 `x_0(<b) + x_t(b)`
   - 得到当前 block 的 logits
   - 更新 `x_t(b)`
4. 当前 block 达到停止条件后，得到最终 `x_0(b)`。
5. 把 `x_0(b)` 正式提交到 history，进入下一个 block。

这就是 block-wise denoise，只不过不再追求 token-level exact cache 对齐，而是追求 block-level self-consistency。

## 5. 为什么这个近似对 SSM 更自然

因为 `SSM / FLA` 擅长的是：

- 给定一个前缀
- 从左到右递推状态
- 输出最后的压缩状态

它不擅长：

- 复杂二维 attention mask
- token 级随机访问
- 改一个 token 只局部修一点后续 state

所以对它更自然的接口是：

1. 从 block boundary 的历史状态开始。
2. 在当前 block 上完整 rollout。
3. 如果当前 block 内容变了，就从这个 boundary 重新 rollout 一次。

这正好对应我们的近似 BD：

- `block checkpoint`
- `whole-block rerollout`
- `finalize-on-commit`

## 6. Attention 和 SSM 各自怎么处理

### 6.0 当前 HF 实现下的关键校正

在当前 Hugging Face `Qwen3NextModel.forward()` 里，模型入口已经先把两类 mask 分开了：

- `causal_mask` 给 `full_attention`
- `linear_attn_mask` 给 `linear_attention`

也就是说，MVP 的补丁点不该是“在 decoder layer 里硬分流同一个 mask”，而应该是：

1. 在模型前向入口准备两份 mask。
2. `full_attention` 层吃自定义 `bd_mask_4d`。
3. `DeltaNet / linear_attention` 层继续吃 `padding_mask_2d`。

这点很重要，因为 `DeltaNet` 这边仍然把 mask 当 padding mask 语义使用，不该把 4D BD mask 硬塞进去。

### 6.1 Attention 分支

Attention 这边可以更“像 BD 一点”，因为它天然支持：

- history cache
- 当前 block 局部重算
- block 级缓存提交

最简单的实现方式是：

- 训练时直接吃 `x_0(<b) + x_t(b)` 这个前缀输入
- full-attention 层吃一张“裁剪版 BD” 的 4D additive mask
- 这张 mask 只描述 `x_0(<b) + x_t(b)` 这个子图，而不是 dFactory 里的双倍长度 `[x_t | x_0]`
- 当前 block 内部双向可见，同时对全部 clean prefix 可见

一个关键实现点：

- 不要全局把模型切到 “non-causal decoder” 模式
- 更稳的做法是保持模型默认配置不动，只给 full-attention 层传入准备好的 4D mask
- 在当前 `transformers` 里，准备好的 4D mask 会直接被 `create_causal_mask()` 原样返回

所以 attention 分支的 MVP 不是“全模型改成双向”，而是“只对 full-attention 层喂 reduced BD mask”。

### 6.2 SSM / FLA 分支

SSM 这边不要硬追 token-level exactness，直接接受 block 级近似。

具体做法：

1. 在每个 block boundary 保存一次 `history_state`。
2. 当前 block 每轮 refine：
   - 从该 boundary 的 `history_state` 出发
   - 用当前版本的 `x_t(b)` 重新跑完整个 block
   - 得到新的当前 block 输出
3. 在 block finalize 前，不把这轮 draft state 写回 history。
4. block finalize 后，才用最终 `x_0(b)` 更新 history state。

这里的关键词只有三个：

- `snapshot`
- `rerollout`
- `commit`

## 7. 近似的本质：牺牲什么，保留什么

### 7.1 牺牲的东西

1. 不能保证 token 级严格等价于 original BD。
2. 不能做到“改一个 token，只修补一小段 SSM state”。
3. 一些理论上的 exactness 和并行效率会丢。

### 7.2 保留的东西

1. 宏观上仍然是 block-wise generation。
2. 历史块仍然是 clean，当前块仍然是 noisy/draft。
3. 训练和推理仍然遵守同一种近似语义。
4. state 不会被中间 draft 污染。

所以这不是乱搞，而是：

**在 hybrid 架构约束下，保住 block-level BD 语义，放弃 token-level exactness。**

## 8. 训练实现建议

### 8.1 不要全量前缀展开

如果一个样本有很多 block，真按下面这样全做：

- `x_t(1)`
- `x_0(1) + x_t(2)`
- `x_0(1) + x_0(2) + x_t(3)`
- ...

复杂度接近平方增长，训练会很重。

### 8.2 更现实的训练办法

每个 step 只采一个或少数几个 block：

1. 随机采样当前训练 block `b`
2. 构造输入 `x_0(<b) + x_t(b)`
3. 只对当前 block 计算 loss

如果上下文太长，再进一步截断历史：

- `x_0(b-K ... b-1) + x_t(b)`

这会损失一部分长历史条件，但训练成本能压住。

### 8.3 Loss 定义

loss 只打在当前 block 里被 mask 的位置。

也就是：

- 历史 clean prefix 不算 loss
- 当前 block 未被 mask 的位置不算 loss
- 当前 block 被 mask 的位置算 CE loss

必要时可以加一个“draft consistency”辅助项，但第一版不建议多加花活。

## 9. 推理实现建议

推理语义上可以理解为维护两类状态：

1. `history cache/state`
   - attention 的 KV cache
   - SSM 的 recurrent/conv state
   - 只包含 finalize 后的 clean blocks

2. `draft block`
   - 当前正在 denoise 的 block token
   - 每轮 refine 可能变化

推理循环：

1. 取当前 block 边界的 `history cache/state`
2. 当前 block 初始化为全 `mask`
3. 做若干轮 refine：
   - 从同一个 boundary snapshot 出发
   - 跑 `history + current draft block`
   - 更新 draft block
4. block finalize
5. 用最终 `x_0(b)` 更新历史 cache/state
6. 进入下一 block

关键纪律：

**每轮 refine 都从同一个 boundary snapshot 出发，而不是接着上一轮的 draft state 往下滚。**

否则状态会脏。

但工程上，**第一版实现不建议立刻上 cache**。原因很简单：

- attention cache 和 DeltaNet cache 都有自己的接口假设
- Qwen-Next 当前 chunk 路径对多 token block 的状态续跑并不是现成为 BD 设计的

所以 MVP 更稳的落地是：

1. `use_cache=False`
2. 每轮 refine 都把 `clean prefix + current draft block` 整段重新前向一次
3. 只更新当前 block
4. 等主干调通后，再做 block-boundary snapshot / commit 优化

也就是说：

- `history/draft 隔离` 仍然是方法目标
- `第一版不碰 cache` 是工程简化手段

## 10. 为什么这条路值得试

虽然这不是 exact BD，但它有三个现实优点：

1. 不要求 `SSM` 学复杂二维 mask。
2. 和 `SSM` 的 prefix-recurrence 本性一致。
3. 训练和推理都能统一成 `clean prefix + noisy current block` 这个接口。

对 `Qwen-Next` 这种 hybrid 模型，这比强行把所有层都塞进 `[x_t | x_0] + custom mask` 更现实。

## 11. 主要风险

1. 训练成本会明显高于单次 forward 的 pure-attention BD。
2. block 太小，rerollout 次数太多，吞吐会难看。
3. block 太大，当前块 denoise 难度会上去。
4. hybrid 融合处仍可能出现“attention 更像 BD、SSM 更像 prefix LM”的张力。
5. 如果 history/draft 隔离做不好，bug 会非常脏，而且不容易查。

## 12. 最小可行版本（MVP）

建议从最小版本开始：

1. 固定 block size。
2. 每步只采一个 block 做训练。
3. 训练输入只用 `clean prefix + noisy current block`。
4. 数据侧保留真 `[MASK]` token，当前 block 按 `sigma` 随机做 masking。
5. labels 只保留当前 block 的 masked positions，loss 用 same-token sparse CE，不做 next-token shift。
6. 模型入口显式准备两份 mask：
   - `padding_mask_2d`
   - `bd_mask_4d`
7. `full_attention` 只吃 `bd_mask_4d`，而且先固定走 eager/additive mask 路径。
8. `DeltaNet / linear_attention` 只吃 `padding_mask_2d`，不去伪装成二维 BD。
9. 第一版训练和推理都不要依赖 `ForCausalLM.forward()` 里的现成 loss，直接取 logits 自己算 sparse CE。
10. 第一版推理 `use_cache=False`，每轮 refine 整段重跑 `clean prefix + current draft block`。
11. 先不追求 fancy 的 token editing、MBE、RL，也先不追求 block-boundary cache 优化。

先把这条主干调通，再考虑更复杂的编辑机制。

## 13. 一句话总结

**Qwen-Next 上做 Block Diffusion，不该强逼 SSM 去学 pure-attention 那套 `[x_t|x_0] + 2D mask`。更合理的做法是把 BD 改写成“clean history prefix + noisy current block”的多次前向近似版本：full-attention 吃 reduced 4D BD mask，DeltaNet 保持 prefix recurrence，第一版整段重跑，不碰 cache 黑魔法。**
