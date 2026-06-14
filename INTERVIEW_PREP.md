# TripleSwitch-LLM — 面试准备指南

> 配合 `PROJECT_ARCHITECTURE.md`（技术原理）和 `model.py`（代码实现）一起学习。
> 按优先级排列：⭐ = 核心必问，⭐⭐ = 常见追问，⭐⭐⭐ = 加分项

---

## 一、两分钟项目自我介绍（背熟）

> **面试官你好，这是我的一个开源项目 TripleSwitch-LLM。**
> 
> 它是一个大模型架构验证框架，核心是三个正交的切换维度——**Attention**（MHA/GQA/MQA/MLA）、**FFN**（FFN/MoE）、**残差连接**（Pre-LN/MHC），一共可以组合出 **16 种架构配置**，通过命令行参数一键切换。
> 
> 技术上，我重点实现了三个东西：
> 1. **MLA（Multi-head Latent Attention）** — 来自 DeepSeek-V2，通过低秩压缩把 KV Cache 降为 MHA 的 5%，并实现了矩阵吸收来加速推理
> 2. **MoE（Mixture of Experts）** — 带负载均衡辅助损失的稀疏门控
> 3. **MHC（Manifold HyperConnection）** — 来自 DeepSeek-V4，用 Sinkhorn 归一化的双随机矩阵做跨层混合
> 
> 另外还有完整的训练管线：预训练 → SFT → LoRA → DPO → 蒸馏。整个项目基于 PyTorch 和 HuggingFace 实现。

---

## 二、核心面试题：MLA（⭐⭐ 必问必会）

### 2.1 MLA 解决了什么问题？

**问题：** 传统的 MHA（Multi-Head Attention）在推理时需要缓存每个 head 的 K 和 V，KV Cache 大小 = 2 × n_heads × head_dim × seq_len × n_layers。对于长上下文场景（比如 32K、128K token），KV Cache 成为显存瓶颈，限制了推理的 batch size。

**MLA 的解决思路：** 把 KV 压缩到一个低维的 latent 空间，而不是直接缓存完整的 K 和 V。

### 2.2 MLA 的核心公式（用手在白板上画出来）

```
c_t^KV = W^{DKV} h_t           # KV 压缩：[hidden] → [kv_lora_rank]
k_t^C  = W^{UK} c_t^KV         # K content 重构
v_t^C  = W^{UV} c_t^KV         # V 重构
q_t^C  = W^{UQ} W^{DQ} h_t     # Q 低秩分解
q_t^R  = W^{QR} h_t            # Q 解耦 RoPE
k_t^R  = W^{KR} h_t            # K 共享 RoPE（所有 head 共享同一个！）
```

**面试话术：**
> "MLA 的核心创新是把 KV 压缩到 kv_lora_rank 维的 latent space。推理时只需要缓存 c_t^KV 和 k_t^R，而不是完整的 K 和 V。以我们的配置为例——hidden_size=512, n_heads=8, head_dim=64, kv_lora_rank=32, rope_dim=16——MHA 每层每 token 需要缓存 2×8×64=1024 个值，而 MLA 只需要缓存 32+16=48 个，节省了 95%。"

### 2.3 矩阵吸收（Matrix Absorption）⭐

**面试话术：**
> "矩阵吸收是 MLA 推理加速的关键。它解决的问题是：如果推理时也显式构造 K，那低秩压缩就没意义了——还是需要把 c_kv 投影到 n_heads×qk_nope 维。所以我们在初始化时预计算了吸收矩阵 w_absorbed[h] = q_b_proj[h]^T @ k_b_proj[h]，推理时直接计算 q_latent^T @ w_absorbed[h] @ c_kv，跳过了显式的 K 展开。"

**关键点（面试官可能追问）：**
- **训练用显式路径**：因为需要梯度回传到 q_b_proj 和 k_b_proj
- **推理用吸收路径**：w_absorbed 是 register_buffer，不参与梯度计算
- **w_absorbed 不会自动更新**：如果模型权重变了需要重新计算——但推理时 model.eval() 冻结了权重，所以没问题

### 2.4 解耦 RoPE（Decoupled RoPE）⭐

**面试话术：**
> "标准的 RoPE 会破坏低秩结构——因为 RoPE 是旋转操作，会让 KV 在旋转后无法被低秩表示。MLA 的解法是把 RoPE 从 content 中分离出来：content 部分做低秩压缩和吸收，RoPE 部分单独用一个小的投影矩阵 W^{KR} 计算，而且 K 的 RoPE 是所有 head 共享的，只需要 rope_dim 维度的缓存。"

### 2.5 KV Cache 对比表（记住数字）

| 变体 | 每层每 token 缓存 | 8层×32K序列的总缓存 |
|------|------------------|-------------------|
| MHA | 2×8×64 = **1024** | 8×32K×1024 = **256MB** |
| GQA(2) | 2×2×64 = **256** | 64MB |
| MQA | 2×1×64 = **128** | 32MB |
| **MLA** | 32+16 = **48** | **12MB** |

---

## 三、核心面试题：MoE（⭐⭐ 必问）

### 3.1 MoE 的基本结构

```
输入 → 门控路由器 → Top-K 选择 → 路由到专家 → 加权求和
                                                    ↑
                                          共享专家（全部 token 都过）
```

### 3.2 负载均衡辅助损失（Load Balancing Loss）⭐

**面试话术：**
> "MoE 有一个经典问题：路由器可能会倾向于把大部分 token 都路由到同一个"偷懒"的专家。所以我们加了辅助损失来做负载均衡。具体来说，我们统计每个专家被选中的频率 f_i 和平均门控分数 P_i，然后计算 aux_loss = α × Σ(f_i × P_i)。当某个专家被选中过多时，f_i 增大，损失也会增大，梯度会推动路由器把 token 分散给其他专家。"

**两种方式（面试官可能问区别）：**
- **Sequence-level（seq_aux=True）**：在整个序列维度统计频率，更稳定
- **Token-level**：每个 token 单独统计，更细粒度但可能波动大

### 3.3 训练 vs 推理的差异

**训练：** 循环遍历每个专家，用 mask 筛选属于该专家的 token。PyTorch 自动追踪梯度。

**推理：** 按 expert 排序，批处理。先对 topk_idx 排序，然后 `bincount` 统计每个专家的 token 数，分段计算。

---

## 四、核心面试题：MHC（⭐⭐⭐ 加分项）

### 4.1 MHC 解决了什么问题？

**面试话术：**
> "标准的残差连接是 h = sublayer(norm(h)) + h，每一层独立处理。MHC 的思路是让各层之间可以有信息交流——用一个可学习的 L×L 矩阵对 所有层的输出做加权混合，约束为双随机矩阵（行和列都归一化）。这样每一层的输出不仅是当前层的计算结果，还可以融合其他层的信息。"

### 4.2 Sinkhorn 归一化

```
weight = softmax(weight, dim=-1)   # 行归一化
weight = softmax(weight, dim=-2)   # 列归一化
# 交替迭代 3 次，最终得到双随机矩阵
```

**面试话术：**
> "Sinkhorn 归一化通过对行和列交替做 softmax，把任意矩阵约束为双随机矩阵。行归一化确保每层的权重之和为 1，列归一化确保所有层对某一层的"贡献"之和也为 1。迭代 3 次就能收敛到不错的近似。"

---

## 五、常见追问（⭐⭐）

### 5.1 Pre-LN 和 Post-LN 的区别？

> "Pre-LN 把 LayerNorm 放在子层之前，梯度更稳定，训练更顺。Post-LN 把 LayerNorm 放在残差之后，是原始 Transformer 的设计，但深层时梯度容易消失。现在的大模型基本都是 Pre-LN。"

### 5.2 为什么用 RMSNorm 而不是 LayerNorm？

> "RMSNorm 只计算均方根，不计算均值，省了约 5-10% 的计算量。LLaMA、DeepSeek 等主流模型都用的 RMSNorm。"

### 5.3 SwiGLU 为什么比 ReLU 好？

> "SwiGLU = SiLU(gate) × up，是一种门控机制。相比 ReLU，SwiGLU 的梯度更平滑，而且门控结构让网络可以有更丰富的表达能力。缺点是多了一个投影矩阵，参数量大了约 50%。"

### 5.4 权重共享（Weight Tying）的好处？

> "Embedding 层和 lm_head 共享权重矩阵，参数量减少了 vocab_size × hidden_size。同时推理时 embedding 和 lm_head 的向量空间是对齐的，理论上对生成质量也有帮助。"

### 5.5 你的项目对比了不同架构的实验结果吗？

**如果没跑过实验（诚实版）：**
> "目前我设计了完整的对比方案——MLA vs MHA 的 KV Cache 对比、MoE vs FFN 的参数量和推理速度对比——但因为资源限制（单卡），还没跑完完整的实验。我在文档里列出了详细的对比维度和预期结果，计划在有 GPU 资源后完成验证。"

---

## 六、针对实习面试的加分策略

### 6.1 让人印象深刻的回答方式

❌ 不要背定义：
> "MLA 是 Multi-head Latent Attention..."

✅ 要说"为什么"：
> "我实现 MLA 是因为 DeepSeek-V2 证明了 KV Cache 可以压缩 95% 而不损失效果，这对推理部署意义很大..."

### 6.2 展示"你做了工程决策"

- "我选择用 register_buffer 存 w_absorbed 而不是每次 forward 都重新算，因为推理时权重是冻结的"
- "我把训练和推理分成两条路径，因为训练需要梯度回传，推理不需要"
- "MoE 的推理我用批处理而不是循环，因为 for 循环逐个专家在 PyTorch 里很慢"

### 6.3 展示"你理解局限性"

- "MLA 的 V 吸收到 O 我还没完全优化，目前推理还是显式重构了 V，这里可以进一步加速"
- "RoPE 目前也没有完全吸收进矩阵，因为 rope_dim=16 很小，直接算的开销不大，但如果有时间可以优化"
- "Flash Attention 还没接入，计划用 xformers 替换手动 attention"

---

## 七、快速记忆卡

| 概念 | 一句话解释 |
|------|-----------|
| **MLA** | KV 压缩到低维 latent + 解耦 RoPE + 矩阵吸收推理加速 |
| **矩阵吸收** | 推理时把 K 的上投影吸收到 Q，跳过显式 K 构造 |
| **解耦 RoPE** | RoPE 从 content 分离，K 的 RoPE 所有 head 共享 |
| **MoE 路由** | Top-K 选择专家 + 加权求和 |
| **aux_loss** | α × Σ(f_i × P_i)，防止 token 扎堆同一个专家 |
| **Sinkhorn** | 交替行/列 softmax，把矩阵约束为双随机 |
| **MHC** | L×L 双随机矩阵混合所有层输出，取最后一行 |
| **GQA** | Q 多组共享同一组 KV，折中 MHA 和 MQA |
| **RoPE** | 绝对位置编码实现相对位置感知，通过旋转矩阵 |

---

## 八、模板问题回答

### Q: 这个项目中你遇到的最大挑战是什么？

> "最大的挑战是 MLA 的矩阵吸收实现。一开始我不理解为什么训练和推理要用两套不同的计算路径——后来明白是因为训练需要梯度回传到 q_b_proj 和 k_b_proj 两个投影矩阵，而推理时 w_absorbed 是这俩矩阵的乘积，一旦冻结就不需要再算了。我卡了好几天才把吸收矩阵的初始化和 forward 里的分支逻辑写对。"

### Q: 这个项目是你一个人做的吗？

> "是的，从代码实现到文档都是我完成的。项目参考了 DeepSeek-V2/V3/V4 的论文和开源实现，但核心代码——MLA 的矩阵吸收、MoE 的负载均衡、MHC 的 Sinkhorn 归一化——是我根据论文公式手写的。"

### Q: 你了解当前大模型的最新进展吗？

> "了解一些。我关注了 DeepSeek 的系列工作——V2 的 MLA、V3 的 MoE 训练优化、V4 的 MHC。另外 LLaMA 系列从 1 到 3 的演进（Post-LN → Pre-LN → GQA → SwiGLU → RoPE）也是我设计三个切换维度的参考。"

---

*面试前把 PROJECT_ARCHITECTURE.md 第 7、8 节（MLA 和 MoE 深度解析）再过一遍，把 model.py 中 MLA.forward 和 MoEGate.forward 的代码读通。祝你面试顺利！*
