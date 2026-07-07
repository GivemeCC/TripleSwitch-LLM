"""
TripleSwitch-LLM: 三重切换架构大模型训练框架

三个正交的架构切换维度：
  - Attention: MHA / GQA / MQA / MLA
  - FFN: FFN / MoE
  - 残差连接: 标准 Pre-LN / MHC

共 4 × 2 × 2 = 16 种架构组合。
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast


# ============================================================
# TripleConfig
# ============================================================

class TripleConfig(PretrainedConfig):
    """
    TripleSwitch 模型配置。
    继承 HuggingFace PretrainedConfig，支持 AutoModel 加载。
    """
    model_type = "triple_switch"

    def __init__(
        self,
        hidden_size: int = 512,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        num_hidden_layers: int = 8,
        intermediate_size: Optional[int] = None,
        vocab_size: int = 6400,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1000000.0,
        dropout: float = 0.0,
        flash_attn: bool = True,

        # MLA 参数
        use_mla: bool = False,
        kv_lora_rank: int = 32,
        q_lora_rank: int = 256,
        rope_dim: int = 16,

        # MoE 参数
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,

        # MHC 参数
        use_mhc: bool = False,

        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.flash_attn = flash_attn

        # MLA
        self.use_mla = use_mla
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_dim = rope_dim

        # MoE
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob

        # MHC
        self.use_mhc = use_mhc


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization。
    相比 LayerNorm 不计算均值，开销更低。
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在 float32 下计算归一化，再转回原精度
        input_dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight.float() * x).type(input_dtype)


# ============================================================
# RoPE（Rotary Position Embedding）
# ============================================================

def precompute_freqs_cis(
    dim: int, max_position: int, theta: float = 1000000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 的 cos/sin 值。
    freqs = outer(m, 1/(theta^(range(0,dim,2)/dim)))
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_position)
    freqs = torch.outer(t, freqs)  # [max_len, dim/2]
    freqs_cos = freqs.cos().repeat(1, 2)  # 复制到满 dim
    freqs_sin = freqs.sin().repeat(1, 2)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 Q 和 K 应用 RoPE 旋转位置编码。
    """
    # cos/sin: [1, seq_len, dim], q/k: [batch, seq_len, heads, dim]
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
    sin = sin.unsqueeze(0).unsqueeze(2)

    # rotate_half: [-x[..., d/2:], x[..., :d/2]]
    q_embed = q * cos + torch.cat([-q[..., q.size(-1)//2:], q[..., :q.size(-1)//2]], dim=-1) * sin
    k_embed = k * cos + torch.cat([-k[..., k.size(-1)//2:], k[..., :k.size(-1)//2]], dim=-1) * sin
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    扩展 KV 头数以匹配 Q 头数（GQA/MQA 需要）。
    [batch, seq, n_kv_heads, dim] -> [batch, seq, n_heads, dim]
    """
    if n_rep == 1:
        return x
    batch, seq, n_kv_heads, dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch, seq, n_kv_heads, n_rep, dim)
        .reshape(batch, seq, n_kv_heads * n_rep, dim)
    )


# ============================================================
# Attention（MHA / GQA / MQA 统一实现）
# ============================================================

class Attention(nn.Module):
    """
    多头注意力机制，支持 MHA / GQA / MQA。
    - MHA: num_key_value_heads == num_attention_heads
    - GQA: num_key_value_heads < num_attention_heads（分组查询）
    - MQA: num_key_value_heads == 1（多查询）
    """
    def __init__(self, config: TripleConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len, _ = x.shape
        cos, sin = pos_emb

        # 线性投影
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # RoPE
        q, k = apply_rotary_pos_emb(q, k, cos[:, :seq_len, :], sin[:, :seq_len, :])

        # KV Cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        past_kv = (k, v) if use_cache else None

        # GQA: 扩展 KV 头
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # 注意力计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_kv


# ============================================================
# MLA（Multi-head Latent Attention）
# ============================================================

class MLA(nn.Module):
    """
    Multi-head Latent Attention（DeepSeek-V2 完整实现）。

    核心创新：
    1. **低秩 KV 压缩**：将 KV 压缩到低维 latent 空间，大幅减少 KV Cache
    2. **解耦 RoPE**：将 RoPE 从 content 中分离，通过共享的 K RoPE 避免 RoPE 破坏低秩结构
    3. **矩阵吸收**：推理时将 W^{UK} 吸收到 Q 投影，跳过显式 K 构造

    KV Cache 对比（hidden_size=512, n_heads=8, kv_lora_rank=32）：
    - MHA: 2 × 8 × 64 = 1024 每层每 token
    - GQA(2): 2 × 2 × 64 = 256
    - MQA: 2 × 1 × 64 = 128
    - MLA: 32 + 16 = 48（节省 95% vs MHA）
    """
    def __init__(self, config: TripleConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads  # MLA 中未使用，保留接口兼容
        self.head_dim = self.hidden_size // self.n_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.rope_dim = config.rope_dim
        self.qk_nope_head_dim = self.head_dim - self.rope_dim

        # Q 低秩分解：W^{DQ} W^{UQ}
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)   # W^{DQ}
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_nope_head_dim, bias=False)  # W^{UQ}

        # Q 解耦 RoPE：W^{QR}
        self.q_rope_proj = nn.Linear(self.hidden_size, self.n_heads * self.rope_dim, bias=False)  # W^{QR}

        # KV 下投影：W^{DKV}
        self.kv_a_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)  # W^{DKV}

        # K content 上投影：W^{UK}
        self.k_b_proj = nn.Linear(self.kv_lora_rank, self.n_heads * self.qk_nope_head_dim, bias=False)

        # V 上投影：W^{UV}
        self.v_b_proj = nn.Linear(self.kv_lora_rank, self.n_heads * self.head_dim, bias=False)

        # K 共享 RoPE：W^{KR}（所有 head 共享同一个 k_pe）
        self.k_rope_proj = nn.Linear(self.hidden_size, self.rope_dim, bias=False)  # W^{KR}

        # O 投影
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

        # 矩阵吸收（推理使用，不可训练）
        # w_absorbed[h] = q_b_proj[h]^T @ k_b_proj[h]  # [q_lora_rank, kv_lora_rank]
        # o_absorbed = W^O @ W^{UV}  # [hidden_size, kv_lora_rank]
        self.register_buffer("w_absorbed", None)
        self.register_buffer("o_absorbed", None)

        self._init_absorbed_weights()

    def _init_absorbed_weights(self):
        """
        预计算吸收矩阵用于推理时的矩阵吸收加速。

        K 吸收到 Q：
            score_nope[h] = q_latent^T @ w_absorbed[h] @ c_kv
            w_absorbed[h] = q_b_proj[h]^T @ k_b_proj[h]  # [q_lora_rank, kv_lora_rank]

        V 吸收到 O（已预计算，当前推理仍用显式 V）：
            o_absorbed = W^O @ W^{UV}  # [hidden_size, kv_lora_rank]
        """
        # q_b_proj: [q_lora_rank, n_heads * qk_nope] -> 拆成 n_heads 个 [q_lora_rank, qk_nope]
        q_b = self.q_b_proj.weight.view(self.n_heads, self.q_lora_rank, self.qk_nope_head_dim)
        # k_b_proj: [kv_lora_rank, n_heads * qk_nope] -> 拆成 n_heads 个 [kv_lora_rank, qk_nope]
        k_b = self.k_b_proj.weight.view(self.n_heads, self.kv_lora_rank, self.qk_nope_head_dim)
        # w_absorbed[h] = q_b[h] @ k_b[h].T  # [q_lora_rank, kv_lora_rank]
        w_absorbed = torch.einsum('hqd,hkd->hqk', q_b, k_b)
        self.w_absorbed = w_absorbed.contiguous()

        # V 吸收到 O（可选）
        # o_proj: [hidden_size, n_heads * head_dim]
        # v_b_proj: [kv_lora_rank, n_heads * head_dim]
        # o_absorbed = W^O @ W^{UV}  # [hidden_size, kv_lora_rank]
        o_proj_w = self.o_proj.weight  # [hidden_size, n_heads * head_dim]
        v_b_w = self.v_b_proj.weight   # [kv_lora_rank, n_heads * head_dim]
        self.o_absorbed = torch.mm(o_proj_w, v_b_w.T)  # [hidden_size, kv_lora_rank]

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len, _ = x.shape
        cos, sin = pos_emb  # MLA 专用 RoPE（rope_dim 维度）

        # ---- Q 路径 ----
        q_latent = self.q_a_proj(x)  # [batch, seq, q_lora_rank]
        q_nope = self.q_b_proj(q_latent).view(batch, seq_len, self.n_heads, self.qk_nope_head_dim)
        q_pe = self.q_rope_proj(x).view(batch, seq_len, self.n_heads, self.rope_dim)

        # ---- KV 路径 ----
        c_kv = self.kv_a_proj(x)  # [batch, seq, kv_lora_rank]

        # 训练 vs 推理双路径
        if past_key_value is None and self.training:
            # ---- 训练：显式构造 K/V，所有参数都有梯度 ----
            k_nope = self.k_b_proj(c_kv).view(batch, seq_len, self.n_heads, self.qk_nope_head_dim)
            v = self.v_b_proj(c_kv).view(batch, seq_len, self.n_heads, self.head_dim)

            # K 共享 RoPE（所有 head 共享同一个 k_pe）
            k_pe = self.k_rope_proj(x).view(batch, seq_len, 1, self.rope_dim)  # [b, s, 1, rope]
            k_pe_expand = k_pe.expand(-1, -1, self.n_heads, -1)

            q = torch.cat([q_nope, q_pe], dim=-1)  # [b, s, n_heads, head_dim]
            k = torch.cat([k_nope, k_pe_expand], dim=-1)

            # RoPE（对 q_pe / k_pe 部分应用）
            q_embed = q * cos + torch.cat([-q[..., q.size(-1)//2:], q[..., :q.size(-1)//2]], dim=-1) * sin
            k_embed = k * cos + torch.cat([-k[..., k.size(-1)//2:], k[..., :k.size(-1)//2]], dim=-1) * sin

            past_kv = None
            present_kv = (c_kv, k_pe.squeeze(2)) if use_cache else None

            scores = torch.matmul(q_embed, k_embed.transpose(-2, -1)) / math.sqrt(self.head_dim)

        else:
            # ---- 推理：矩阵吸收，不构造 K ----
            # Content 部分：K 吸收到 Q
            q_absorb = torch.einsum('bsq,hqk->bhsk', q_latent, self.w_absorbed)  # [b, n_heads, s, kv_lora]
            content_scores = torch.matmul(q_absorb, c_kv.unsqueeze(-1)).squeeze(-1)  # [b, n_heads, s]

            # RoPE 部分
            q_pe_emb = q_pe * cos + torch.cat([-q_pe[..., q_pe.size(-1)//2:], q_pe[..., :q_pe.size(-1)//2]], dim=-1) * sin

            if past_key_value is not None:
                past_c_kv, past_k_pe = past_key_value
                c_kv = torch.cat([past_c_kv, c_kv], dim=1)
                k_pe = torch.cat([past_k_pe, self.k_rope_proj(x).view(batch, seq_len, self.rope_dim)], dim=1)
            else:
                k_pe = self.k_rope_proj(x).view(batch, seq_len, self.rope_dim)

            k_pe_emb = k_pe * cos + torch.cat([-k_pe[..., k_pe.size(-1)//2:], k_pe[..., :k_pe.size(-1)//2]], dim=-1) * sin

            past_kv = (c_kv, k_pe) if use_cache else None

            # 拼接 content scores 和 rope scores
            seq_len_kv = c_kv.size(1)
            rope_scores = torch.matmul(
                q_pe_emb.view(batch, seq_len, self.n_heads, self.rope_dim),
                k_pe_emb.view(batch, 1, seq_len_kv, self.rope_dim).transpose(-2, -1)
            ).squeeze(-1)  # [b, n_heads, s, kv_len] squeeze 最后一维

            # 合并注意力分数
            scores = content_scores.view(batch, self.n_heads, seq_len, 1)
            rope_scores = rope_scores.view(batch, self.n_heads, seq_len, seq_len_kv)
            scores = (scores + rope_scores) / math.sqrt(self.head_dim)

            # 重构 V 用于注意力加权
            v = self.v_b_proj(c_kv).view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)  # [b, n_heads, kv_len, head_dim]

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(x)

        if past_key_value is not None or not self.training:
            # 推理路径：需要手动做 attention @ v
            attn_output = torch.matmul(attn_weights, v)  # [b, n_heads, s, head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        else:
            # 训练路径：attn_weights @ v
            v_training = self.v_b_proj(c_kv).view(batch, seq_len, self.n_heads, self.head_dim)
            v_training = v_training.transpose(1, 2)  # [b, n_heads, s, head_dim]
            attn_output = torch.matmul(attn_weights, v_training)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)

        attn_output = self.o_proj(attn_output)
        return attn_output, past_kv


# ============================================================
# FeedForward（SwiGLU）
# ============================================================

class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络。
    output = down_proj(up_proj(x) * silu(gate_proj(x)))
    """
    def __init__(self, config: TripleConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
            config.intermediate_size = intermediate_size
        else:
            intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================
# MoE（Mixture of Experts）
# ============================================================

class MoEGate(nn.Module):
    """
    MoE 门控路由：top-k 专家选择 + 负载均衡辅助损失。

    Sequence-level 辅助损失（seq_aux=True）：
        aux_loss = α * Σ_i (f_i · P_i)
        其中 f_i 是专家 i 被选中的频率，P_i 是专家 i 的平均门控分数
    """
    def __init__(self, config: TripleConfig):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.scoring_func = config.scoring_func
        self.aux_loss_alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, config.hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)  # [total_tokens, hidden]
        total_tokens = hidden_states.size(0)

        # 门控分数
        logits = F.linear(hidden_states, self.weight)
        if self.scoring_func == 'softmax':
            scores = F.softmax(logits.float(), dim=-1).type_as(hidden_states)
        else:
            scores = F.sigmoid(logits)

        # Top-k 选择
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        # 辅助损失（负载均衡）
        if self.seq_aux:
            # Sequence-level
            ones = torch.ones_like(topk_idx)
            ce = torch.zeros(total_tokens, self.n_routed_experts, device=hidden_states.device)
            ce.scatter_add_(1, topk_idx, ones)
            ce = ce / (total_tokens * self.top_k / self.n_routed_experts)
            mean_score = scores.mean(0)
            aux_loss = (ce * mean_score).sum() * self.aux_loss_alpha
        else:
            # Token-level
            ce = F.one_hot(topk_idx, num_classes=self.n_routed_experts).float().mean(1)
            Pi = scores.mean(0, keepdim=True)
            fi = ce * self.n_routed_experts
            aux_loss = (Pi * fi).sum() * self.aux_loss_alpha

        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x  # 做 skip connection
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制专家的选择
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # 对每个token，复制 num_experts_per_tok 多份，
            # 这样做的目的是为了将每个token同时传入其top-K个被选中的专家里面进行计算
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # 创建一个与x形状相同但是类型为 float16 的空张量，用于存储每个token经过对应专家处理后的结果
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                # flat_topk_idx 是一个索引张量，表示每个token被分配给了哪个专家
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            # 将输出按照token和专家维度重新组织
            # 使用 topk_weight 权重对每个专家的输出进行加权求和
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # 把最终输出恢复成原始输入的形状
            y = y.view(*orig_shape)
        else:
            # 在推理阶段使用更高效的函数 moe_infer 处理 MOE 部分
            # 通常是为了减少内存冗余或计算冗余，例如合并多个token，一起处理
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 如果启用了共享专家，它们会作用在所有的token上
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        # 通常这个损失会加到 total_loss = task_loss + config.aux_loss_coeff * model.aux_loss
        self.aux_loss = aux_loss

        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # tokens_per_expert = [6, 15, 20, 26] 这四个数值分别代表4个专家处理的token数量
        tokens_idxs = idxs // self.config.num_experts_per_tok
        # token_idxs = [3, 7, 19, 21, 24, 25, 4, 5, 6, 10, 11, 12...] 代表着 token_idxs[:6]
        # 属于0号专家的；每个token有可能被多个专家处理，取决于 config.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前专家处理token的起始索引
            start_idx = 0 if i==0 else tokens_per_expert[i-1]
            # 如果没有token被分配给这个专家，跳过该专家
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = tokens_idxs[start_idx:end_idx]
            # 从原始的输入x中获取这些token的嵌入
            expert_tokens = x[exp_token_idx]
            # 输入到当前专家网络中进行前向传播；
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 对专家输出进行加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 将专家输出加到最终的输出张量上面去，加权之后的求和
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


# ============================================================
# MHC（Manifold HyperConnection）
# ============================================================

class MHCMixer(nn.Module):
    """
    流形约束跨层混合。
    维护一个 [L, L] 的可学习权重矩阵，通过 Sinkhorn 归一化
    （行 softmax → 列 softmax 迭代）约束为双随机矩阵。

    取混合结果的最后一行作为最终输出。
    """
    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.mix_weight = nn.Parameter(torch.randn(num_layers, num_layers) * 0.01)

    def _sinkhorn_normalize(self, x: torch.Tensor, n_iter: int = 3) -> torch.Tensor:
        """
        Sinkhorn 归一化：交替对行和列做 softmax。
        将矩阵约束为双随机（行和列分别归一化）。
        """
        for _ in range(n_iter):
            x = F.softmax(x, dim=-1)  # 行归一化
            x = F.softmax(x, dim=-2)  # 列归一化
        return x

    def forward(self, layer_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        layer_outputs: [L, batch, seq, hidden]
        returns: [batch, seq, hidden] — 混合后的最后一层输出
        """
        # Sinkhorn 归一化混合权重
        weight = self._sinkhorn_normalize(self.mix_weight)  # [L, L]

        # 堆叠所有层输出: [L, batch, seq, hidden]
        stacked = torch.stack(layer_outputs, dim=0)

        # 加权混合: [L, batch, seq, hidden] @ [L, L]^T = [batch, seq, hidden]
        mixed = torch.einsum('lbsh,hl->bsh', stacked, weight)
        return mixed


# ============================================================
# TripleBlock
# ============================================================

class TripleBlock(nn.Module):
    """
    TripleSwitch Transformer Block。

    可切换组件：
    - Attention: MLA 或 标准 Attention（MHA/GQA/MQA）
    - FFN: MoE 或 标准 FFN

    采用 Pre-LN 残差结构：
        residual = x
        x = self_attn(norm(x)) + residual
        residual = x
        x = mlp(norm(x)) + residual
    """
    def __init__(self, layer_id: int, config: TripleConfig):
        super().__init__()
        self.layer_id = layer_id

        if config.use_mla:
            self.self_attn = MLA(config)
        else:
            self.self_attn = Attention(config)

        if config.use_moe:
            self.mlp = MOEFeedForward(config)
        else:
            self.mlp = FeedForward(config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        x = self.input_layernorm(x)
        x, present_kv = self.self_attn(
            x, pos_emb, attention_mask, past_key_value, use_cache
        )
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual

        return x, present_kv


# ============================================================
# TripleModel
# ============================================================

class TripleModel(nn.Module):
    """
    TripleSwitch 模型主体：多层 Transformer 堆叠。
    支持 RoPE 预计算和可选 MHC 跨层混合。
    """
    def __init__(self, config: TripleConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            TripleBlock(i, config) for i in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MHC 跨层混合器
        if config.use_mhc:
            self.mhc = MHCMixer(config.num_hidden_layers, config.hidden_size)
        else:
            self.mhc = None

        # 预计算 RoPE
        if config.use_mla:
            # MLA 使用 rope_dim 维度的 RoPE
            head_dim = config.rope_dim
        else:
            head_dim = config.hidden_size // config.num_attention_heads

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=head_dim,
            max_position=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        batch, seq_len = input_ids.shape

        # Token embedding
        h = self.embed_tokens(input_ids)

        # RoPE position embeddings
        cos = self.freqs_cos[:seq_len].unsqueeze(0)  # [1, seq, dim]
        sin = self.freqs_sin[:seq_len].unsqueeze(0)
        pos_emb = (cos, sin)

        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device),
                diagonal=1
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        else:
            causal_mask = attention_mask

        # 逐层前向
        presents = []
        layer_outputs = [] if self.mhc is not None else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None and i < len(past_key_values) else None
            h, present = layer(h, pos_emb, causal_mask, past_kv, use_cache)
            if use_cache:
                presents.append(present)
            if layer_outputs is not None:
                layer_outputs.append(h)

        # MHC 跨层混合
        if self.mhc is not None and layer_outputs is not None:
            h = self.mhc(layer_outputs)

        h = self.norm(h)

        # MoE 辅助损失汇总
        aux_loss = sum(
            layer.mlp.aux_loss for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return h, presents, aux_loss


# ============================================================
# TripleModelForCausalLM
# ============================================================

class TripleModelForCausalLM(PreTrainedModel, GenerationMixin):
    """
    TripleSwitch 因果语言模型。
    继承 HuggingFace PreTrainedModel + GenerationMixin，支持 transformers 生态。
    """
    config_class = TripleConfig

    def __init__(self, config: TripleConfig):
        super().__init__(config)
        self.model = TripleModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重共享（embedding ↔ lm_head）
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
    ) -> CausalLMOutputWithPast:
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if logits_to_keep > 0:
            logits = self.lm_head(h[:, -logits_to_keep:, :])
        else:
            logits = self.lm_head(h)

        return CausalLMOutputWithPast(
            last_hidden_state=h,
            logits=logits,
            aux_loss=aux_loss,
            past_key_values=past_kvs if use_cache else None,
        )
