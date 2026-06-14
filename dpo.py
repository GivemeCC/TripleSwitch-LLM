import os
import argparse
import time
import math
import warnings

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer

from model import TripleConfig, TripleModelForCausalLM
from dataset import DPODataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def logits_to_probs(logits, labels):
    """将 logits 转为每个 token 的对数概率（log_softmax + gather）。"""
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


def dpo_loss(ref_probs, probs, mask, beta):
    """
    DPO 损失函数：
    -log(sigmoid(β * (π_θ(y_w|x) - π_ref(y_w|x) - π_θ(y_l|x) + π_ref(y_l|x))))
    """
    seq_lengths = mask.sum(dim=1, keepdim=True)
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze()

    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]
    rejected_ref_probs = ref_probs[batch_size // 2:]
    chosen_probs = probs[:batch_size // 2]
    rejected_probs = probs[batch_size // 2:]

    pi_logratios = chosen_probs - rejected_probs
    ref_logratios = chosen_ref_probs - rejected_ref_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/')
    model = TripleModelForCausalLM(lm_config)

    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'LLM 可训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    # 参考模型（冻结）
    ref_model = TripleModelForCausalLM(lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.requires_grad_(False)

    model = model.to(args.device)
    ref_model = ref_model.to(args.device)
    return model, ref_model, tokenizer


def init_distributed_model():
    if not ddp:
        return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


def train_epoch(epoch):
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        x_chosen = batch['x_chosen'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(current_step=epoch * iter_per_epoch + step,
                    total_steps=args.epochs * iter_per_epoch, lr=args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_probs = logits_to_probs(ref_logits, y)
            ref_probs = ref_probs * mask

            outputs = model(x)
            logits = outputs.logits
            probs = logits_to_probs(logits, y)
            probs = probs * mask

            loss = dpo_loss(ref_probs, probs, mask, beta=0.1)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1, args.epochs, step, iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                )
            )

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/dpo_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TripleSwitch DPO")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default='bfloat16')
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--use_moe", default=False, type=bool)
    parser.add_argument("--use_mla", default=False, type=bool)
    parser.add_argument("--kv_lora_rank", default=32, type=int)
    parser.add_argument("--q_lora_rank", default=256, type=int)
    parser.add_argument("--rope_dim", default=16, type=int)
    parser.add_argument("--use_mhc", default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./data/sft_mini_512.jsonl")
    args = parser.parse_args()

    lm_config = TripleConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe, use_mla=args.use_mla, kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=args.q_lora_rank, rope_dim=args.rope_dim, use_mhc=args.use_mhc,
    )

    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 42
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_model()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    model, ref_model, tokenizer = init_model(lm_config)
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, pin_memory=True,
        drop_last=False, shuffle=False, num_workers=args.num_workers,
        sampler=train_sampler,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)
