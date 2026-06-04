from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), None)


class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached[:, :, :seq_len, :].to(dtype=dtype), self._sin_cached[:, :, :seq_len, :].to(dtype=dtype)

    def at_position(self, position: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        cos, sin = self.forward(position + 1, device, dtype)
        return cos[:, :, position : position + 1, :], sin[:, :, position : position + 1, :]


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.ones(1, num_heads, 1, 1, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))

    def forward_step(
        self,
        x: Tensor,
        cache: tuple[Tensor, Tensor] | None,
        position: int,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        bsz, seqlen, dim = x.shape
        if seqlen != 1:
            raise ValueError("forward_step expects exactly one token")
        q = self.c_q(x).reshape(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary.at_position(position, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)
        if cache is None:
            all_k, all_v = k, v
        else:
            prev_k, prev_v = cache
            all_k = torch.cat((prev_k, k), dim=2)
            all_v = torch.cat((prev_v, v), dim=2)
        y = F.scaled_dot_product_attention(
            q,
            all_k,
            all_v,
            is_causal=False,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        out = self.proj(y.transpose(1, 2).contiguous().reshape(bsz, 1, dim))
        return out, (all_k, all_v)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(1, dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(1, dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(1, dim), torch.zeros(1, dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, :, :] * x + mix[1][None, :, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, :, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, :, :] * self.mlp(self.mlp_norm(x))
        return x

    def forward_step(
        self,
        x: Tensor,
        x0: Tensor,
        cache: tuple[Tensor, Tensor] | None,
        position: int,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, :, :] * x + mix[1][None, :, :] * x0
        attn_out, cache = self.attn.forward_step(self.attn_norm(x), cache, position)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, :, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, :, :] * self.mlp(self.mlp_norm(x))
        return x, cache


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1024,
        num_layers: int = 9,
        model_dim: int = 512,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        mlp_mult: int = 2,
        logit_softcap: float = 30.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, 1, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm()

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, :, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward_logits_step(
        self,
        input_id: Tensor,
        caches: list[tuple[Tensor, Tensor] | None] | None = None,
        position: int = 0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        if input_id.ndim == 0:
            input_id = input_id[None]
        if input_id.ndim != 1:
            raise ValueError("input_id must be a scalar or 1D tensor")
        if caches is None:
            caches = [None] * len(self.blocks)
        if len(caches) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} cache entries, got {len(caches)}")

        x = self.tok_emb(input_id[:, None])
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        new_caches: list[tuple[Tensor, Tensor]] = []
        for i in range(self.num_encoder_layers):
            x, cache = self.blocks[i].forward_step(x, x0, caches[i], position)
            new_caches.append(cache)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            block_idx = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, :, :] * skips.pop()
            x, cache = self.blocks[block_idx].forward_step(x, x0, caches[block_idx], position)
            new_caches.append(cache)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits[:, -1], new_caches


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_baseline_model(checkpoint_path: str | Path, device: torch.device | None = None) -> GPT:
    device = device or default_device()
    model = GPT()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    return model
