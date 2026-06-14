# TripleSwitch-LLM

**三重切换架构的大模型训练框架** — Attention × FFN × 残差连接，共 16 种架构一键切换。

## 核心特性

三个正交的架构切换维度，覆盖 Transformer 演进的三个核心方向：

| 维度 | 选项 | 技术来源 |
|------|------|---------|
| Attention | **MHA** → **GQA** → **MQA** → **MLA** | Transformer → LLaMA2 → LLaMA3 → DeepSeek-V2 |
| FFN | **FFN** → **MoE** | 密集 → 稀疏（Mixtral / DeepSeek-V3） |
| 残差连接 | **Pre-LN** → **MHC** | 标准 → DeepSeek-V4 |

任意组合，一条命令切换：

```bash
# MHA + FFN（最小配置）
python pretrain.py

# MLA + MoE（DeepSeek-V2 风格）
python pretrain.py --use_mla --kv_lora_rank 32 --use_moe

# 全开：MLA + MoE + MHC
python pretrain.py --use_mla --use_moe --use_mhc
```

## 技术亮点

- **MLA（Multi-head Latent Attention）** — DeepSeek-V2 完整实现，含低秩 KV 压缩、解耦 RoPE、**矩阵吸收**（训练显式 / 推理吸收双路径），KV Cache 节省 95% vs MHA
- **MoE（Mixture of Experts）** — Top-k 路由 + sequence-level 负载均衡辅助损失 + 共享专家
- **MHC（Manifold HyperConnection）** — Sinkhorn 归一化双随机矩阵约束的跨层混合
- **LoRA** — Monkey-Patch 注入的低秩适配
- **完整训练管线** — Pretrain → Full SFT → LoRA SFT → DPO → Distillation

## 快速开始

```bash
# 安装依赖
pip install torch transformers

# 预训练（MLA + MoE）
python pretrain.py --use_mla --use_moe --hidden_size 512

# 全量微调
python full_sft.py --use_mla --hidden_size 512

# LoRA 微调
python lora_sft.py --use_mla --lora_name my_lora

# DPO 对齐
python dpo.py --use_mla

# 知识蒸馏（学生 512/8 → 教师 768/16）
python distillation.py --use_mla
```

## 项目结构

```
├── model.py            # 核心模型：Config, Attention, MLA, MoE, MHC, CausalLM
├── dataset.py          # 数据集：Pretrain / SFT / DPO
├── model_lora.py       # LoRA 低秩适配
├── pretrain.py         # 预训练
├── full_sft.py         # 全量微调
├── lora_sft.py         # LoRA 微调
├── dpo.py              # DPO 偏好对齐
├── distillation.py     # 知识蒸馏
├── convert_model.py    # HuggingFace 格式转换
├── tokenizer.json      # BPE 分词器 (vocab_size=6400)
├── tokenizer_config.json
└── model/              # tokenizer 加载目录
```

## 架构矩阵

| Attention | FFN | 残差 | 命令行 |
|-----------|-----|------|--------|
| MHA | FFN | Pre-LN | 默认 |
| GQA | FFN | Pre-LN | `--num_key_value_heads 4` |
| MQA | MoE | Pre-LN | `--num_key_value_heads 1 --use_moe` |
| MLA | MoE | MHC | `--use_mla --use_moe --use_mhc` |

## License

MIT
