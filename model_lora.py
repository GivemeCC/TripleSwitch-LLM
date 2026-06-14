import torch
from torch import nn


class LoRA(nn.Module):
    """LoRA 低秩适配层：A(高斯) @ B(全零)，初始时输出为 0。"""
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    """通过 Monkey-Patch 将 LoRA 注入所有方阵 Linear 层。"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
                rank=rank,
            ).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):
    """加载 LoRA 权重。"""
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {
                k.replace(f'{name}.lora.', ''): v
                for k, v in state_dict.items() if f'{name}.lora.' in k
            }
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """保存 LoRA 权重。"""
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {
                f'{name}.lora.{k}': v
                for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
    torch.save(state_dict, path)
