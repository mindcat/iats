"""Two-stage toki pona MT finetune with atomic <tp:WORD> tokens.

Stage 1: transformer frozen, only the new <tp:*>/<cartouche> embedding rows train
         (they snap into the base manifold without touching English fluency).
Stage 2: LoRA on every attention/MLP projection + the new rows.

    uv run python train_mt.py --model Qwen/Qwen2.5-0.5B-Instruct --out runs/qwen05
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import mt_data
from align_loss import AttentionProbe, example_alignment, load_links
from init_embeddings import warm_start

HERE = os.path.dirname(os.path.abspath(__file__))
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ------------------------------------------------------------------ data

def encode_all(encoder, examples, cache_path, links=None):
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    rows = []
    for ex in examples:
        r = encoder.encode(ex)
        if r is None:
            continue
        if links is not None:
            r["align"] = example_alignment(encoder, ex, links, r["input_ids"])
        rows.append(r)
    torch.save(rows, cache_path)
    return rows


def make_batches(rows, token_budget, rng):
    """Length-bucketed batches whose padded size stays under token_budget."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]) + rng.random())
    batches, cur, cur_max = [], [], 0
    for i in order:
        n = len(rows[i]["input_ids"])
        if cur and max(cur_max, n) * (len(cur) + 1) > token_budget:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, n)
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(rows, idx, pad_id, device):
    n = max(len(rows[i]["input_ids"]) for i in idx)
    ids = torch.full((len(idx), n), pad_id, dtype=torch.long)
    labels = torch.full((len(idx), n), -100, dtype=torch.long)
    attn = torch.zeros((len(idx), n), dtype=torch.long)
    for b, i in enumerate(idx):
        r = rows[i]
        ids[b, :len(r["input_ids"])] = torch.tensor(r["input_ids"])
        labels[b, :len(r["labels"])] = torch.tensor(r["labels"])
        attn[b, :len(r["input_ids"])] = 1
    return ids.to(device), labels.to(device), attn.to(device)


# ------------------------------------------------------------------ model

def decoder_and_head(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.get_output_embeddings()


def lm_loss(model, ids, labels, attn):
    """Next-token loss with logits computed only where there is a target (also runs the probe hook)."""
    decoder, head = decoder_and_head(model)
    hidden = decoder(input_ids=ids, attention_mask=attn).last_hidden_state
    tgt = labels[:, 1:]
    mask = tgt != -100
    logits = head(hidden[:, :-1][mask]).float()
    return F.cross_entropy(logits, tgt[mask])


def new_row_mask(emb, new_ids):
    mask = torch.zeros(emb.shape[0], 1, device=emb.device, dtype=emb.dtype)
    mask[new_ids] = 1
    return mask


@torch.no_grad()
def dev_loss(model, rows, batches, pad_id, device, limit=60):
    model.eval()
    total, n = 0.0, 0
    for idx in batches[:limit]:
        ids, labels, attn = collate(rows, idx, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total += lm_loss(model, ids, labels, attn).item()
        n += 1
    model.train()
    return total / max(n, 1)


def run_stage(name, model, params, rows, dev_rows, args, steps, pad_id, device, rng, log, probe=None):
    opt = torch.optim.AdamW(params, weight_decay=0.0, betas=(0.9, 0.98))
    warmup = min(100, steps // 10)
    base_lrs = [g["lr"] for g in opt.param_groups]
    dev_batches = make_batches(dev_rows, args.token_budget, random.Random(0))
    step, t0, running, running_align, n_align = 0, time.time(), 0.0, 0.0, 0
    model.train()
    while step < steps:
        for idx in make_batches(rows, args.token_budget, rng):
            if step >= steps:
                break
            scale = (step + 1) / warmup if step < warmup else \
                0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * scale
            ids, labels, attn = collate(rows, idx, pad_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lm_loss(model, ids, labels, attn)
                align = probe.loss([rows[i].get("align") for i in idx]) if probe else None
            total = loss + args.align_lambda * align if align is not None else loss
            total.backward()
            if align is not None:
                running_align += align.item()
                n_align += 1
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            running += loss.item()
            step += 1
            if step % args.log_every == 0:
                msg = {"stage": name, "step": step, "of": steps, "loss": running / args.log_every,
                       "lr": opt.param_groups[0]["lr"], "elapsed_min": (time.time() - t0) / 60}
                if n_align:
                    msg["align_kl"] = running_align / n_align
                running, running_align, n_align = 0.0, 0.0, 0
                if step % args.eval_every == 0 or step == steps:
                    msg["dev_loss"] = dev_loss(model, dev_rows, dev_batches, pad_id, device)
                log(msg)
    return dev_loss(model, dev_rows, dev_batches, pad_id, device)


def save(model, tp, args, out, extra):
    os.makedirs(out, exist_ok=True)
    emb = model.get_input_embeddings().weight
    new_ids = tp.tp_ids + [tp.cart_open_id, tp.cart_close_id]
    rows = {"ids": new_ids, "embed": emb[new_ids].detach().cpu()}
    out_emb = model.get_output_embeddings().weight
    if out_emb.data_ptr() != emb.data_ptr():
        rows["lm_head"] = out_emb[new_ids].detach().cpu()
    torch.save(rows, os.path.join(out, "tp_rows.pt"))
    if hasattr(model, "save_pretrained") and hasattr(model, "peft_config"):
        model.save_pretrained(os.path.join(out, "adapter"))
    tp.base.save_pretrained(os.path.join(out, "tokenizer"))
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump({"args": vars(args), "tp_offset": tp.tp_offset, "n_tp": len(tp.words), **extra}, f, indent=1)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage1-steps", type=int, default=1000)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--emb-lr-stage1", type=float, default=1e-3)
    ap.add_argument("--emb-lr-stage2", type=float, default=1e-4)
    ap.add_argument("--token-budget", type=int, default=12288)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--max-examples", type=int, default=0, help="subsample training examples (0 = all)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--no-warm-start", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (needed for >=1.5B)")
    ap.add_argument("--align-lambda", type=float, default=0.0, help="guided alignment loss weight (plan: 0.1)")
    ap.add_argument("--align-layer", type=float, default=0.7, help="supervised layer as a fraction of depth")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    log_f = open(os.path.join(args.out, "log.jsonl"), "a")

    def log(msg):
        print(json.dumps(msg), flush=True)
        log_f.write(json.dumps(msg) + "\n")
        log_f.flush()

    tok = AutoTokenizer.from_pretrained(args.model)
    encoder = mt_data.ChatEncoder(tok, max_len=args.max_len)
    tp = encoder.tp
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    cache = os.path.join(HERE, "cache")
    os.makedirs(cache, exist_ok=True)
    tag = args.model.replace("/", "__")
    links = load_links() if args.align_lambda else None
    train_rows = encode_all(encoder, mt_data.build_examples("train", seed=args.seed),
                            os.path.join(cache, f"{tag}.train{'.align' if links else ''}.pt"), links)
    dev_rows = encode_all(encoder, mt_data.build_examples("dev", seed=args.seed),
                          os.path.join(cache, f"{tag}.dev.pt"))
    if args.max_examples:
        train_rows = rng.sample(train_rows, min(args.max_examples, len(train_rows)))
    n_tokens = sum(len(r["input_ids"]) for r in train_rows)
    log({"train_examples": len(train_rows), "train_tokens": n_tokens, "dev_examples": len(dev_rows)})

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if len(tok) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok), mean_resizing=False)
    new_ids = tp.tp_ids + [tp.cart_open_id, tp.cart_close_id]
    refined = False
    if not args.no_warm_start:
        refined = warm_start(model, tp, extra_ids=[(tp.cart_open_id, ' "'), (tp.cart_close_id, '"')])
    log({"warm_start": not args.no_warm_start, "w2v_ridge_refined": refined})

    # embeddings: only the new rows may move
    emb = model.get_input_embeddings().weight
    out_emb = model.get_output_embeddings().weight
    emb_params = [emb] if out_emb.data_ptr() == emb.data_ptr() else [emb, out_emb]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in emb_params:
        p.requires_grad_(True)
        mask = new_row_mask(p, new_ids)
        p.register_hook(lambda g, m=mask: g * m)

    batches_per_epoch = len(make_batches(train_rows, args.token_budget, random.Random(0)))
    run_stage("embed", model, [{"params": emb_params, "lr": args.emb_lr_stage1}], train_rows, dev_rows,
              args, args.stage1_steps, pad_id, device, rng, log)

    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    for p in emb_params:
        p.requires_grad_(True)
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    stage2_steps = int(args.epochs * batches_per_epoch)
    log({"stage2_steps": stage2_steps, "lora_params": sum(p.numel() for p in lora_params)})
    probe = AttentionProbe(model, args.align_layer) if args.align_lambda else None
    if probe:
        log({"align_layer": probe.layer_index,
             "supervised_examples": sum(1 for r in train_rows if r.get("align"))})
    final_dev = run_stage("lora", model, [{"params": lora_params, "lr": args.lr},
                                          {"params": emb_params, "lr": args.emb_lr_stage2}],
                          train_rows, dev_rows, args, stage2_steps, pad_id, device, rng, log, probe)
    save(model, tp, args, args.out, {"final_dev_loss": final_dev})
    log({"saved": args.out, "final_dev_loss": final_dev})


if __name__ == "__main__":
    main()
