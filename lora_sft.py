import os
import argparse
import time
import math
import warnings

import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer

from model import TripleConfig, TripleModelForCausalLM
from model_lora import save_lora, load_lora, apply_lora
from dataset import SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/')
    model = TripleModelForCausalLM(lm_config)

    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'LLM 可训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model, tokenizer


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
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(current_step=epoch * iter_per_epoch + step,
                    total_steps=args.epochs * iter_per_epoch, lr=args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
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
            lora_save_path = f'{args.save_dir}/lora/{args.lora_name}_{lm_config.hidden_size}.pth'
            os.makedirs(os.path.dirname(lora_save_path), exist_ok=True)
            save_lora(model, lora_save_path)
            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TripleSwitch SFT with LoRA")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--epochs", type=int, default=10)
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
    parser.add_argument("--data_path", type=str, default="./data/lora_medical.jsonl")
    parser.add_argument("--lora_name", type=str, default="lora_medical")
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

    model, tokenizer = init_model(lm_config)
    apply_lora(model)

    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    if not ddp or dist.get_rank() == 0:
        print(f"总参数量: {total_params}")
        print(f"LoRA 参数量: {lora_params_count}")
        print(f"LoRA 占比: {lora_params_count / total_params * 100:.2f}%")

    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    lora_params = [param for name, param in model.named_parameters() if 'lora' in name]

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, pin_memory=True,
        drop_last=False, shuffle=False, num_workers=args.num_workers,
        sampler=train_sampler,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)
