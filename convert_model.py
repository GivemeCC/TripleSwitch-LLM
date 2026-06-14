import os

import torch
import warnings
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from model import TripleConfig, TripleModelForCausalLM

warnings.filterwarnings('ignore', category=UserWarning)


def convert_torch2transformers_triple(torch_path, transformers_path, dtype=torch.bfloat16):
    """
    将 TripleSwitch checkpoint 保存为 HuggingFace TripleModelForCausalLM 格式。
    """
    TripleConfig.register_for_auto_class()
    TripleModelForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    lm_model = TripleModelForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)

    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数量: {model_params / 1e6:.2f}M = {model_params / 1e9:.3f}B')

    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained("./model/")
    tokenizer.save_pretrained(transformers_path)
    print(f'模型已保存为 TripleSwitch 格式: {transformers_path}')


def convert_torch2transformers_llama(torch_path, transformers_path, dtype=torch.bfloat16):
    """
    将 TripleSwitch checkpoint 转换为 HuggingFace LlamaForCausalLM 格式，
    兼容第三方生态。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    llama_config = LlamaConfig(
        vocab_size=lm_config.vocab_size,
        hidden_size=lm_config.hidden_size,
        intermediate_size=64 * ((int(lm_config.hidden_size * 8 / 3) + 64 - 1) // 64),
        num_hidden_layers=lm_config.num_hidden_layers,
        num_attention_heads=lm_config.num_attention_heads,
        num_key_value_heads=lm_config.num_key_value_heads,
        max_position_embeddings=lm_config.max_seq_len,
        rms_norm_eps=lm_config.rms_norm_eps,
        rope_theta=lm_config.rope_theta,
    )
    llama_model = LlamaForCausalLM(llama_config)
    llama_model.load_state_dict(state_dict, strict=False)
    llama_model = llama_model.to(dtype)

    model_params = sum(p.numel() for p in llama_model.parameters() if p.requires_grad)
    print(f'模型参数量: {model_params / 1e6:.2f}M = {model_params / 1e9:.3f}B')

    llama_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained("./model/")
    tokenizer.save_pretrained(transformers_path)
    print(f'模型已保存为 LlamaForCausalLM 格式: {transformers_path}')


if __name__ == "__main__":
    lm_config = TripleConfig(
        hidden_size=512, num_hidden_layers=8,
        max_seq_len=512, use_moe=True, use_mla=False,
    )

    torch_path = './out/pretrain_512_moe.pth'
    transformers_path = './converted_model'

    convert_torch2transformers_triple(
        torch_path=torch_path, transformers_path=transformers_path,
    )

    # convert_torch2transformers_llama(
    #     torch_path=torch_path, transformers_path=transformers_path,
    # )
