"""Export-friendly Qwen2 decoder with a stateful KV cache and an attention-linker output.

Pure torch + safetensors (no transformers), so it traces under the coremltools torch pin.

forward(input_ids (1, q), causal_mask (1, 1, q, end), source_mask (1, end))
  -> logits for the last position (1, 1, V)
  -> alignment (1, end): last query's attention summed over heads of the linker layers
     (tp_vocab.json "align_layers", written by merge.py; default [0.6L, 0.8L)), restricted to source-word positions and renormalized (AttentiveTranslatorWrapper)

The cache occupies positions [end - q, end); prefill and decode share one graph.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype) * self.weight


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h, kv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        hd = d // h
        self.input_layernorm = RMSNorm(d, cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(d, cfg["rms_norm_eps"])
        self.q_proj = nn.Linear(d, h * hd, bias=True)
        self.k_proj = nn.Linear(d, kv * hd, bias=True)
        self.v_proj = nn.Linear(d, kv * hd, bias=True)
        self.o_proj = nn.Linear(h * hd, d, bias=False)
        self.gate_proj = nn.Linear(d, cfg["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(d, cfg["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(cfg["intermediate_size"], d, bias=False)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Qwen2Export(nn.Module):
    def __init__(self, cfg, max_context=512, align_layers=None):
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg["num_hidden_layers"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = cfg["hidden_size"] // self.n_heads
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(self.n_layers))
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.tie = cfg.get("tie_word_embeddings", True)
        if not self.tie:
            self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        if align_layers is None:
            align_layers = range(int(0.6 * self.n_layers), int(0.8 * self.n_layers))
        self.align_layers = set(align_layers)
        inv = 1.0 / (cfg.get("rope_theta", 10000.0) ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.qk_scale = self.head_dim ** -0.25
        shape = (self.n_layers, self.n_kv, max_context, self.head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=torch.float16))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=torch.float16))

    @classmethod
    def from_pretrained(cls, path, max_context=512, dtype=torch.float32):
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        align_layers = None
        if os.path.exists(os.path.join(path, "tp_vocab.json")):
            with open(os.path.join(path, "tp_vocab.json")) as f:
                align_layers = json.load(f).get("align_layers")
        model = cls(cfg, max_context, align_layers)
        state = {}
        for name in sorted(os.listdir(path)):
            if name.endswith(".safetensors"):
                state.update(load_file(os.path.join(path, name)))
        mapped = {}
        for k, v in state.items():
            k = k.removeprefix("model.")
            k = k.replace("self_attn.", "").replace("mlp.", "")
            mapped[k] = v
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        missing = [m for m in missing if m not in ("k_cache", "v_cache") and not (model.tie and m == "lm_head.weight")]
        unexpected = [u for u in unexpected if not (model.tie and u == "lm_head.weight")]
        assert not missing and not unexpected, (missing, unexpected)
        model.k_cache = model.k_cache.to(torch.float16)
        return model.to(dtype).eval()

    def forward(self, input_ids, causal_mask, source_mask):
        q_len = input_ids.shape[-1]
        end = causal_mask.shape[-1]
        start = end - q_len
        h = self.embed_tokens(input_ids)
        pos = torch.arange(q_len, device=h.device).float() + start
        freqs = pos[:, None] * self.inv_freq[None]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos()[None, None].to(h.dtype), emb.sin()[None, None].to(h.dtype)
        groups = self.n_heads // self.n_kv
        align = torch.zeros(1, end, dtype=h.dtype, device=h.device)

        for i, layer in enumerate(self.layers):
            x = layer.input_layernorm(h)
            q = layer.q_proj(x).view(1, q_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = layer.k_proj(x).view(1, q_len, self.n_kv, self.head_dim).transpose(1, 2)
            v = layer.v_proj(x).view(1, q_len, self.n_kv, self.head_dim).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            # 1/sqrt(d) split across q and k so q.k stays inside fp16 range (Qwen2.5-1.5B overflows otherwise);
            # the cache therefore holds pre-scaled keys
            q, k = q * self.qk_scale, k * self.qk_scale
            self.k_cache[i:i + 1, :, start:end] = k.to(self.k_cache.dtype)
            self.v_cache[i:i + 1, :, start:end] = v.to(self.v_cache.dtype)
            K = self.k_cache[i:i + 1, :, :end].to(h.dtype).repeat_interleave(groups, dim=1)
            V = self.v_cache[i:i + 1, :, :end].to(h.dtype).repeat_interleave(groups, dim=1)
            scores = q @ K.transpose(-1, -2) + causal_mask.to(h.dtype)
            probs = F.softmax(scores.float(), dim=-1).to(h.dtype)
            if i in self.align_layers:
                align = align + probs[:, :, -1].sum(1)
            o = (probs @ V).transpose(1, 2).reshape(1, q_len, self.n_heads * self.head_dim)
            h = h + layer.o_proj(o)
            x = layer.post_attention_layernorm(h)
            h = h + layer.down_proj(F.silu(layer.gate_proj(x)) * layer.up_proj(x))

        h = self.norm(h[:, -1:])
        logits = h @ self.embed_tokens.weight.T if self.tie else self.lm_head(h)
        align = align * source_mask.to(h.dtype)
        align = align / (align.sum(-1, keepdim=True) + 1e-4)
        return logits, align


def causal_mask(q_len, end, dtype=torch.float32):
    start = end - q_len
    m = torch.full((q_len, end), float("-inf"))
    for r in range(q_len):
        m[r, :start + r + 1] = 0
    return m[None, None].to(dtype)
