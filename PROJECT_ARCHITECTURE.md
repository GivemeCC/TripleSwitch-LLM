# TripleSwitch-LLM — 三重切换架构 LLM 训练框架

---

## 目录

1. [项目定位](#1-项目定位)
2. [整体架构](#2-整体架构)
3. [模型架构 (model.py)](#3-模型架构-modelpy)
4. [训练管线](#4-训练管线)
5. [数据集](#5-数据集)
6. [一键切换矩阵](#6-一键切换矩阵)
7. [MLA 深度解析](#7-mla-深度解析)
8. [MOE 深度解析](#8-moe-深度解析)
9. [训练管线详解](#9-训练管线详解)
10. [LoRA (model_lora.py)](#10-lora-model_lorapy)
11. [配置文件与 Tokenizer](#11-配置文件与-tokenizer)
12. [注意事项与常见问题](#12-注意事项与常见问题)

---

## 1. 项目定位

本项目是一个**大模型架构验证框架**，而非一个追求 SOTA 结果的生产级训练管线。核心设计理念是：

> **三个可切换维度**，覆盖 Transformer 架构演进的三个核心方向（Attention、FFN、残差连接），通过统一的配置接口一键切换，对比不同架构的参数量、推理效率、KV Cache 占用。

| 切换维度 | 选项 | 对应技术演进 |
|---------|------|------------|
| Attention | MHA → GQA → MQA → **MLA** | 原始 Transformer → LLaMA2 → LLaMA3 → **DeepSeek-V2** |
| FFN | FFN → **MoE** | 密集 → 稀疏（Mixtral, DeepSeek-V3） |
| 残差连接 | 标准残差 → **MHC** | Pre-LN → **DeepSeek-V4** |

三个维度**完全正交**，可自由组合出 4×2×2 = 16 种架构配置。

---

## 2. 整体架构

```
model.py          ← 模型定义：Config, Attention, MLA, FFN, MoE, Block, CausalLM
├── TripleConfig               所有超参数 + 开关配置
├── RMSNorm                     根均方归一化
├── precompute_freqs_cis        RoPE 预计算
├── apply_rotary_pos_emb        RoPE 应用
├── repeat_kv                   KV Head 扩展（GQA/MHA）
├── Attention                   MHA / GQA / MQA 统一实现
├── MLA                         Multi-head Latent Attention（DeepSeek-V2 完整实现）
├── FeedForward                 标准 SwiGLU FFN
├── MoEGate                     MoE 门控路由 + 辅助损失
├── MOEFeedForward              MoE 前馈网络（训练循环/推理批处理双路径）
├── MHCMixer                    MHC 流形约束跨层混合（双随机矩阵 + Sinkhorn）
├── TripleBlock                Transformer Block（可切换 Attention + FFN + MHC）
├── TripleModel                     多层堆叠 + RoPE 缓冲 + MHC Mixer
└── TripleModelForCausalLM          因果语言模型（HuggingFace 兼容）

dataset.py       ← 数据集
├── PretrainDataset             预训练（纯文本 next-token）
├── SFTDataset                  指令微调（仅计算 assistant 部分 loss）
└── DPODataset                  DPO 偏好对齐（chosen/rejected 配对）

model_lora.py   ← LoRA 高效微调
├── LoRA                        低秩适配层
├── apply_lora                  注入 LoRA（Monkey-Patch）
├── load_lora                   加载 LoRA 权重
└── save_lora                   保存 LoRA 权重

pretrain.py      ← 预训练管线
full_sft.py      ← 全量微调管线
lora_sft.py      ← LoRA 微调管线
dpo.py           ← DPO 偏好对齐管线
distillation.py  ← 知识蒸馏管线
convert_model.py ← 模型格式转换（→ HuggingFace / LLaMA）
```

---

## 3. 模型架构 (model.py)

### 3.1 TripleConfig

所有模型配置在 `TripleConfig` 中声明，继承自 HuggingFace `PretrainedConfig`。

**基础参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 512 | 隐藏层维度 |
| `num_attention_heads` | 8 | Q 头数 |
| `num_key_value_heads` | 2 | KV 头数（MHA=8, GQA=2/4, MQA=1） |
| `num_hidden_layers` | 8 | Transformer 层数 |
| `intermediate_size` | None | FFN 中间维度（None 时自动计算） |
| `vocab_size` | 6400 | 词表大小 |
| `max_position_embeddings` | 32768 | 最大序列长度 |
| `rms_norm_eps` | 1e-5 | RMSNorm epsilon |
| `rope_theta` | 1000000.0 | RoPE base frequency |
| `dropout` | 0.0 | Dropout 比例 |
| `flash_attn` | True | Flash Attention 开关 |

**MLA 参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_mla` | False | 启用 MLA |
| `kv_lora_rank` | 32 | KV 低秩压缩维度 |
| `q_lora_rank` | 256 | Q 低秩压缩维度 |
| `rope_dim` | 16 | 解耦 RoPE 维度 |

**MoE 参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_moe` | False | 启用 MoE |
| `num_experts_per_tok` | 2 | 每 token 路由的专家数 (top-k) |
| `n_routed_experts` | 4 | 总专家数 |
| `n_shared_experts` | 1 | 共享专家数 |
| `scoring_func` | 'softmax' | 门控评分函数 |
| `aux_loss_alpha` | 0.1 | 辅助损失权重 |
| `seq_aux` | True | 序列级别辅助损失 |
| `norm_topk_prob` | True | 归一化 top-k 权重 |

### 3.2 RMSNorm

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot g
$$

相比 LayerNorm 不计算均值，计算更轻量。训练时通过 `x.float()` 计算归一化，再 `type_as(x)` 恢复原精度。

### 3.3 RoPE

```
freqs = outer(m, 1/(theta^(range(0,dim,2)/dim)))
freqs_cos = cat([cos(freqs), cos(freqs)])  # 复制到满 dim
```

`apply_rotary_pos_emb(q, k, cos, sin)`：
- `rotate_half(x) = cat([-x[..., d/2:], x[..., :d/2]])`
- `q_embed = q*cos + rotate_half(q)*sin`
- `k_embed = k*cos + rotate_half(k)*sin`

**MLA 使用自己的 `rope_cos/sin`**（按 `rope_dim` 维度预计算），而非 `TripleModel` 中针对 `head_dim` 预计算的那套。

### 3.4 Attention（MHA / GQA / MQA 统一实现）

```python
# 唯一区别：num_key_value_heads 取值
# MHA=8, GQA=2/4, MQA=1

self.q_proj = Linear(d, n_heads * head_dim)
self.k_proj = Linear(d, n_kv_heads * head_dim)  # ← 仅 KV 头不同
self.v_proj = Linear(d, n_kv_heads * head_dim)

# 前向时 repeat_kv 扩展：
n_rep = n_heads // n_kv_heads
# [b,s,n_kv,h] → [b,s,n_kv*n_rep,h]
```

### 3.5 MLA（Multi-head Latent Attention）

参见第 [7 节](#7-mla-深度解析)。

### 3.6 FeedForward（SwiGLU）

```python
output = down_proj(up_proj(x) * silu(gate_proj(x)))
```

`intermediate_size` 自动按 `8/3 * hidden_size` 计算并对齐到 64 的倍数。

### 3.7 MoE

参见第 [8 节](#8-moe-深度解析)。

### 3.8 TripleBlock

```python
class TripleBlock(nn.Module):
    def __init__(self, layer_id, config):
        # Attention 切换
        if config.use_mla:
            self.self_attn = MLA(config)
        else:
            self.self_attn = Attention(config)

        # FFN 切换
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, x, pos_emb, ...):
        # Pre-LN 残差结构
        residual = x
        x = self.self_attn(self.input_layernorm(x), ...) + residual
        x = self.mlp(self.post_attention_layernorm(x)) + x
        return x, present_kv
```

**MHC 模式**：当 `use_mhc=True` 时，`TripleModel.forward` 会收集所有层的输出（每个 Block 内仍正常做 Pre-LN 残差），然后在所有层计算完毕后用 `MHCMixer` 做加权混合。MHCMixer 维护一个 `[L, L]` 的可学习权重矩阵，通过 Sinkhorn 归一化（对行做 softmax → 对列做 softmax）约束为双随机矩阵，最后取混合结果的最后一行作为最终输出输出。

### 3.9 TripleModel

```python
class TripleModel(nn.Module):
    def __init__(self, config):
        self.embed_tokens = Embedding(vocab_size, hidden_size)
        self.layers = ModuleList([TripleBlock(i, config) for _ in range(num_hidden_layers)])
        self.norm = RMSNorm(hidden_size)
        # 预计算 RoPE（标准 Attention 用 head_dim，MLA 用 rope_dim 各自算）
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_len, theta)
        register_buffer("freqs_cos", freqs_cos)
        register_buffer("freqs_sin", freqs_sin)

    def forward(self, input_ids, ...):
        h = embed_tokens(input_ids)
        for layer, past_kv in zip(layers, past_key_values):
            h, present = layer(h, position_embeddings, past_kv, ...)
            presents.append(present)
        h = norm(h)
        aux_loss = sum(l.mlp.aux_loss for l in layers if isinstance(l.mlp, MOEFeedForward))
        return h, presents, aux_loss
```

### 3.10 TripleModelForCausalLM

```python
class TripleModelForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = TripleConfig

    def __init__(self, config):
        self.model = TripleModel(config)
        self.lm_head = Linear(hidden_size, vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight  # 权重共享

    def forward(self, input_ids, ..., logits_to_keep=0):
        h, past_kvs, aux_loss = self.model(...)
        logits = self.lm_head(h[:, -logits_to_keep:, :])  # 只算最后 N 个 logits
        return CausalLMOutputWithPast(
            last_hidden_state=h, logits=logits,
            aux_loss=aux_loss, past_key_values=past_kvs
        )
```

- 继承 `PreTrainedModel` + `GenerationMixin` → HuggingFace 生成兼容
- `logits_to_keep` 用于推理时只计算最后几个位置的 logits（节省显存）
- `self.OUT` 改为每次 forward 新建 `CausalLMOutputWithPast`，消除状态污染

---

## 4. 训练管线

### 4.1 通用结构

所有训练脚本统一模式：

```
parse_args() → init config → init model + tokenizer → init dataloader
→ init optimizer + scaler → loop train_epoch()
```

**共享组件：**

| 组件 | 说明 |
|------|------|
| `Logger()` | 分布式感知打印（非 DDP 或 rank=0 才输出） |
| `get_lr()` | Cosine 调度 `lr/10 + 0.5*lr*(1+cos(π*step/total))` |
| `ctx` | `nullcontext()`(CPU) 或 `torch.cuda.amp.autocast()`(GPU) |
| `GradScaler` | 混合精度缩放 |
| `DistributedDataParallel` | 分布式训练 |

**梯度累积：**
```python
loss = loss / accumulation_steps
scaler.scale(loss).backward()
if (step + 1) % accumulation_steps == 0:
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

- `accumulation_steps=8`：等效 batch_size = 32 × 8 = 256
- 日志中 loss = `loss.item() * accumulation_steps` 恢复真实值

**检查点保存：**
```python
ckp = f'{save_dir}/pretrain_{hidden_size}{"_moe" if use_moe else ""}.pth'
state_dict = {k: v.half() for k, v in model.state_dict().items()}  # FP16 压缩
torch.save(state_dict, ckp)
```

### 4.2 Pretrain（预训练）

```bash
python pretrain.py --use_mla --use_moe --kv_lora_rank 32 --hidden_size 512
```

- 数据：`PretrainDataset`，纯文本 next-token prediction
- 损失：`CE_loss × loss_mask + aux_loss`

### 4.3 Full SFT（全量微调）

```bash
python full_sft.py --use_mla --hidden_size 512
```

- 从 `pretrain_{hidden_size}.pth` 加载 checkpoint
- 全参数更新

### 4.4 LoRA SFT（高效微调）

```bash
python lora_sft.py --use_mla --lora_name lora_medical
```

- 从 `pretrain_{hidden_size}.pth` 加载 checkpoint
- 冻结非 LoRA 参数，仅优化 LoRA 参数

### 4.5 DPO（偏好对齐）

```bash
python dpo.py --use_mla --data_path ./data/dpo_data.jsonl
```

- 从 `full_sft_{hidden_size}.pth` 加载 checkpoint
- 需要参考模型（冻结）+ 训练模型
- 损失：`-log(sigmoid(β * (π_θ(y_w|x) - π_ref(y_w|x) - π_θ(y_l|x) + π_ref(y_l|x))))`

### 4.6 Distillation（知识蒸馏）

```bash
python distillation.py --use_mla
```

- 学生：`hidden_size=512, layers=8`
- 教师：`hidden_size=768, layers=16`
- 损失：`α × CE_loss + (1-α) × KL_distill_loss`

---

## 5. 数据集

### 5.1 PretrainDataset

```python
class PretrainDataset(Dataset):
    def __getitem__(self, index):
        sample = self.samples[index]  # {"text": "..."}
        encoding = tokenizer(sample['text'], max_length=max_len,
                             padding='max_length', truncation=True)
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != pad_token_id)  # padding 不计算 loss
        X = input_ids[:-1]    # <s> i love u
        Y = input_ids[1:]     # i love u </s>
        return X, Y, loss_mask[1:]
```

### 5.2 SFTDataset

- 使用 `apply_chat_template()` 构建对话格式
- `_generate_loss_mask()`：仅在 `assistant` 回复位置置 1

### 5.3 DPODataset

- 格式：`{"chosen": [...], "rejected": [...]}`
- 同时编码 chosen 和 rejected，分别生成 loss_mask
- 返回 dict：`x_chosen/y_chosen/mask_chosen/x_rejected/y_rejected/mask_rejected`

---

## 6. 一键切换矩阵

### 6.1 Attention 切换

| 变体 | 命令行参数 | KV Cache/每层每 token |
|------|-----------|---------------------|
| MHA | 默认（`num_key_value_heads=8`） | `2 × 8 × 64 = 1024` |
| GQA | `--num_key_value_heads 4` | `2 × 4 × 64 = 512` |
| MQA | `--num_key_value_heads 1` | `2 × 1 × 64 = 128` |
| **MLA** | **`--use_mla --kv_lora_rank 32`** | **`32 + 16 = 48`** |

### 6.2 FFN 切换

| 变体 | 命令行参数 |
|------|-----------|
| FFN | 默认 |
| MoE | `--use_moe --n_routed_experts 4` |

### 6.3 残差连接切换

| 变体 | 命令行参数 | 说明 |
|------|-----------|------|
| 标准 Pre-LN | 默认 | `h = sublayer(norm(h)) + h` |
| MHC | `--use_mhc` | 双随机矩阵约束跨层混合 |

### 6.4 组合示例

```bash
# MHA + FFN（最小）
python pretrain.py

# GQA + MoE
python pretrain.py --num_key_value_heads 2 --use_moe

# MLA + MoE + 长上下文
python pretrain.py --use_mla --kv_lora_rank 64 --use_moe --n_routed_experts 8 --max_seq_len 8192

# 全开：MLA + MoE + MHC
python pretrain.py --use_mla --use_moe --use_mhc --n_routed_experts 8

# 低资源 MLA
python pretrain.py --use_mla --kv_lora_rank 16 --q_lora_rank 64 --rope_dim 8
```

---

## 7. MLA 深度解析

### 7.1 DeepSeek-V2 核心公式

```
c_t^KV = W^{DKV} h_t              # KV 压缩 [d → kv_lora_rank]
k_t^C  = W^{UK} c_t^KV            # K content [kv_lora_rank → n_heads × qk_nope]
v_t^C  = W^{UV} c_t^KV            # V [kv_lora_rank → n_heads × head_dim]
q_t^C  = W^{UQ} W^{DQ} h_t        # Q 低秩 [d → q_lora_rank → n_heads × qk_nope]
q_t^R  = W^{QR} h_t               # Q 解耦 RoPE [d → n_heads × rope_dim]
k_t^R  = W^{KR} h_t               # K 解耦 RoPE (共享!) [d → rope_dim]

q_t = [q_t^C; q_t^R]              # [n_heads, head_dim]  head_dim = qk_nope + rope_dim
k_t = [k_t^C; k_t^R]              # k_t^R 在所有 head 间共享!
attention = softmax(q^T k / sqrt(head_dim)) @ v
output = W^O @ attention
```

### 7.2 矩阵吸收

**Content 部分（K 吸收到 Q）：**

```python
# 预计算（init 时一次）：
w_absorbed[h] = q_b_proj[h]^T @ k_b_proj[h]  # [q_lora_rank × kv_lora_rank]

# 训练（显式构造 K，梯度过 q_b_proj / k_b_proj 回传）
q_nope = q_b_proj(q_latent)      # [q_lora_rank → n_heads × qk_nope]
k_nope = k_b_proj(c_kv)          # [kv_lora_rank → n_heads × qk_nope]
score_nope = q_nope^T @ k_nope

# 推理（吸收，不构造 K）
score_nope[h] = q_latent^T @ w_absorbed[h] @ c_kv  # 跳过 n_heads 展开
```

**V 吸收到 O（已预计算，可选）：**

```python
o_absorbed = W^O @ W^{UV}  # [hidden_size × kv_lora_rank]
# output = o_absorbed @ (c_kv @ scores)  ← 可替换显式 V 重构
```

### 7.3 训练/推理双路径

```python
if past_key_value is None and self.training:
    # 训练：显式构造 K，所有参数都有梯度
    q_nope = self.q_b_proj(q_latent).view(...)
    k_nope = self.k_b_proj(c_kv).view(...)
    q = cat([q_nope, q_pe], dim=-1)
    k = cat([k_nope, k_pe_expand], dim=-1)
    scores = q @ k.T / sqrt(head_dim)
else:
    # 推理：矩阵吸收，不构造 K
    q_absorb = einsum('bsq,hqk->bhsk', q_latent, w_absorbed)
    content_scores = matmul(q_absorb, c_kv.T)
    rope_scores = matmul(q_pe, k_pe.T)
    scores = (content_scores + rope_scores) / sqrt(head_dim)
```

### 7.4 KV Cache 对比

| 变体 | 每层每 token 缓存 | 存储量 (8×64, rank=32) |
|------|------------------|----------------------|
| MHA | `(K, V)` 各 8×64 | 1024 |
| GQA(2) | `(K, V)` 各 2×64 | 256 |
| MQA | `(K, V)` 各 1×64 | 128 |
| **MLA** | **`(c_kv, k_pe)` = 32+16** | **48** |

### 7.5 参数量计算

```
Q 低秩:           d×q_lora + q_lora×n_heads×qk_nope   = 512×256 + 256×8×48 = 229,376
Q 解耦PE:         d×n_heads×rope_dim                   = 512×8×16 = 65,536
KV 下投影:        d×kv_lora                            = 512×32 = 16,384
KV K上投影:       kv_lora×n_heads×qk_nope              = 32×8×48 = 12,288
KV V上投影:       kv_lora×n_heads×head_dim             = 32×8×64 = 16,384
K 共享PE:         d×rope_dim                           = 512×16 = 8,192
O 投影:           n_heads×head_dim×d                   = 8×64×512 = 262,144
MLA 合计: ~610K  vs  MHA: ~1,049K  → 节省 42%
```

### 7.6 实现要点

1. **head_dim = qk_nope_head_dim + rope_dim**，典型：64 = 48 + 16
2. **K RoPE 是共享的**：`k_rope_proj` 输出 `rope_dim` 而非 `n_heads × rope_dim`
3. **吸收矩阵不可训练**（`register_buffer`），训练用显式路径，推理用吸收路径
4. **RoPE 在 Cache 之前应用**，确保缓存的 k_pe 已是旋转后的值
5. **Causal Mask 偏移**：`diagonal = 1 + (full_len - seq_len)`，只遮罩当前序列内

---

## 8. MoE 深度解析

### 8.1 MoEGate 门控

```python
self.weight = Parameter(empty(n_routed_experts, hidden_size))

def forward(self, hidden_states):
    logits = F.linear(hidden_states, self.weight)
    scores = softmax(logits)  # [total_tokens, n_experts]
    topk_weight, topk_idx = topk(scores, k=top_k)
    return topk_idx, topk_weight, aux_loss
```

### 8.2 辅助损失

**Sequence-level**（`seq_aux=True`）：
```python
ce.scatter_add_(1, topk_idx, ones) / (seq_len*top_k / n_experts)
aux_loss = (ce * mean_score).sum() * alpha
```

**Token-level**（`seq_aux=False`）：
```python
ce = one_hot(topk_idx).mean(0)   # 专家被选频率
Pi = scores.mean(0)               # 专家平均得分
fi = ce * n_experts
aux_loss = (Pi * fi).sum() * alpha
```

`aux_loss` 由 `TripleModel` 汇总后加到 `total_loss` 中。

### 8.3 MOEFeedForward 前向

**训练**：
```python
x = x.repeat_interleave(top_k, dim=0)        # 每 token 复制
y = zeros_like(x)
for i, expert in enumerate(experts):
    y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])
y = (y.view(topk_weight.shape + [-1]) * topk_weight).sum(dim=1)  # 加权求和
```

**推理**（`moe_infer`）：
```python
idxs = flat_topk_idx.argsort()
tokens_per_expert = bincount().cumsum()
for i, end_idx in enumerate(tokens_per_expert):
    expert = experts[i]
    expert_tokens = x[tokens_idxs[start:end_idx]]
    expert_cache.scatter_add_(0, exp_token_idx, expert_out * weights)
```

### 8.4 共享专家

```python
if n_shared_experts > 0:
    for expert in shared_experts:
        y = y + expert(identity)  # 所有 token 额外过共享专家
```

---

## 9. LoRA (model_lora.py)

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        self.A = Linear(in_features, rank, bias=False)   # N(0, 0.02)
        self.B = Linear(rank, out_features, bias=False)  # zero

    def forward(self, x):
        return self.B(self.A(x))
```

**Monkey-Patch 注入**：
```python
for name, module in model.named_modules():
    if isinstance(module, Linear) and is_square:
        lora = LoRA(...)
        setattr(module, "lora", lora)
        def forward_with_lora(x, layer1=orig_forward, layer2=lora):
            return layer1(x) + layer2(x)
        module.forward = forward_with_lora
```

**保存/加载**：遍历 `named_modules`，筛出含 `lora` 属性的模块。

---

## 10. 模型转换 (convert_model.py)

```bash
python convert_model.py
```

两种输出格式：
1. **`TripleModelForCausalLM`** — 注册 `AutoModelForCausalLM`
2. **`LlamaForCausalLM`** — 直接使用 HuggingFace LLaMA 架构

---

## 11. 配置文件与 Tokenizer

| 属性 | 值 |
|------|-----|
| `tokenizer_class` | `PreTrainedTokenizerFast` |
| `bos_token` | `<|im_start|>` (id=1) |
| `eos_token` | `<|im_end|>` (id=2) |
| `pad_token` | `<|endoftext|>` (id=0) |
| `vocab_size` | 6400 |

对话模板：
```
<|im_start|>system\nYou are a helpful assistant<|im_end|>\n
<|im_start|>user\n{输入}<|im_end|>\n
<|im_start|>assistant\n{回复}<|im_end|>\n
```

---

## 12. 注意事项与常见问题

### 12.1 实现状态

| 特性 | 状态 |
|------|------|
| MHA / GQA / MQA | ✅ |
| MLA + 矩阵吸收 | ✅ |
| MoE + 负载均衡 | ✅ |
| LoRA | ✅ |
| MHC（双随机约束） | ✅ |
| Flash Attention | 🔲 待接入 |
| V 完全吸收到 O | ⚠️ 已预计算但推理仍用显式 V |

### 12.2 已知限制

1. **V 未完全吸收**：虽然预计算了 `o_absorbed`，推理仍从 latent 显式重构 V。替换为 `o_absorbed @ (c_kv @ scores.T)` 可进一步加速。

2. **RoPE 未吸收**：`rope_dim=16` 维度小，直接计算开销不大。如需吸收，将 `q_rope_proj / k_rope_proj` 做类似矩阵乘法融合。

3. **训练/推理路径判断**：当前用 `past_key_value is None and self.training` 区分。无 cache 的 `model.eval()` 推理也走吸收路径（正确但非最优），可简化判断。

4. **`w_absorbed` 不会自动随权重更新**：因为是 `register_buffer`，`optimizer.step()` 改变 `q_b_proj / k_b_proj` 后 `w_absorbed` 不会同步。但训练时用显式路径（梯度正确），推理时 `model.eval()` 冻结权重，无不一致问题。

### 12.3 调试命令

```bash
# 打印参数量
python -c "
from model import TripleConfig, TripleModelForCausalLM
cfg = TripleConfig(hidden_size=512, use_mla=True, kv_lora_rank=32)
m = TripleModelForCausalLM(cfg)
print(f'MLA: {sum(p.numel() for p in m.parameters())/1e6:.2f}M')
"

# 形状冒烟测试
python -c "
import torch; from model import TripleConfig, TripleModelForCausalLM
cfg = TripleConfig(hidden_size=512, use_mla=True, kv_lora_rank=32)
m = TripleModelForCausalLM(cfg)
x = torch.randint(0, 6400, (2, 128))
out = m(x, use_cache=True)
assert out.logits.shape == (2, 128, 6400), f'Shape mismatch: {out.logits.shape}'
assert len(out.past_key_values) == 8
print('✅ Smoke test passed')
"
```
