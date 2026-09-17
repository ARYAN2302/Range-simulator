"""RAPTOR-FM encoder: faithful Radio-FM dual-level attention encoder.

Faithful to Radio-FM (arXiv:2608.05793v1) §III-C:
- intra-channel MHSA over temporal patches (RoPE on Q/K, original positions)
- inter-channel interaction: GAP → 2×2 channel attention α → per-step recalibration
- pre-RMSNorm, LayerScale (γ init 1e-5), progressive DropPath
- NO Perceiver bottleneck
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .raptorfm_tokenizer import RMSNorm, LayerScale, RoPE


class IntraChannelMHSA(nn.Module):
    """Multi-head self-attention over temporal patches for one channel, with RoPE."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope: RoPE):
        """
        x: [B, L, D]
        cos, sin: [L, D] from RoPE
        """
        B, L, D = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,L,Dh]
        k = self.wk(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = rope.apply(q, k, cos, sin)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,L,L]
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # [B,H,L,Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(out)


class InterChannelInteraction(nn.Module):
    """Radio-FM inter-channel recalibration: GAP → 2×2 attention α → per-step update."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

    def forward(self, h_i: torch.Tensor, h_q: torch.Tensor):
        """
        h_i, h_q: [B, L, D] (I and Q channel features, already RMSNorm'd by the caller --
            this module returns the update term ONLY, not a residual-inclusive value, so the
            caller can wrap it in the same pre-norm residual pattern as intra-channel MHSA and
            the FFN: h = h + DropPath(LayerScale(delta)). Returning a residual-inclusive value
            here was a bug: the caller was then adding a second, unnormalized residual on top.
        Returns (delta_i, delta_q): the Σ_c' α_c,c' · (H_c',t W_v) term only.
        """
        # Temporal GAP → channel tokens S ∈ B×2×D
        s_i = h_i.mean(dim=1)  # [B, D]
        s_q = h_q.mean(dim=1)  # [B, D]
        S = torch.stack([s_i, s_q], dim=1)  # [B, 2, D]

        # 2×2 channel attention
        q = self.wq(S)  # [B, 2, D]
        k = self.wk(S)
        # α = softmax((QW_q)(KW_k)^T / √d)
        scale = self.d_model ** -0.5
        alpha = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, 2, 2]
        alpha = alpha.softmax(dim=-1)

        # Per-time-step update: Σ_c' α_c,c' · (H_c',t W_v)
        v_i = self.wv(h_i)  # [B, L, D]
        v_q = self.wv(h_q)

        # α: [B, 2, 2]; v: [B, L, D]
        # For each channel c, sum over c': α[c,c'] * v_c'
        v = torch.stack([v_i, v_q], dim=1)  # [B, 2, L, D]
        # α @ v: [B,2,2] @ [B,2,L,D] → [B,2,L,D]
        update = torch.einsum('bcn,bnld->bcld', alpha, v)  # [B, 2, L, D]

        return update[:, 0], update[:, 1]


class RAPTORFMEncoderBlock(nn.Module):
    """One Radio-FM encoder layer: pre-RMSNorm → intra-MHSA → LayerScale → DropPath
    → inter-channel → LayerScale → FFN → LayerScale."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.intra = IntraChannelMHSA(d_model, n_heads)
        self.ls_intra = LayerScale(d_model)
        self.dp_intra = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm_inter = RMSNorm(d_model)
        self.inter = InterChannelInteraction(d_model)
        self.ls_inter = LayerScale(d_model)
        self.dp_inter = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ls_ffn = LayerScale(d_model)
        self.dp_ffn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, h_i: torch.Tensor, h_q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope: RoPE):
        # Intra-channel MHSA
        h = self.norm1(h_i), self.norm1(h_q)
        attn_i = self.intra(h[0], cos, sin, rope)
        attn_q = self.intra(h[1], cos, sin, rope)
        h_i = h_i + self.dp_intra(self.ls_intra(attn_i))
        h_q = h_q + self.dp_intra(self.ls_intra(attn_q))

        # Inter-channel interaction (pre-norm residual, matching intra/FFN and the paper's
        # Fig 3 -- previously missing its RMSNorm entirely, and double-counting the residual
        # by adding LayerScale(output) to the already-residual-inclusive inter() output)
        h = self.norm_inter(h_i), self.norm_inter(h_q)
        delta_i, delta_q = self.inter(h[0], h[1])
        h_i = h_i + self.dp_inter(self.ls_inter(delta_i))
        h_q = h_q + self.dp_inter(self.ls_inter(delta_q))

        # FFN
        h = self.norm2(h_i), self.norm2(h_q)
        ffn_i = self.ffn(h[0])
        ffn_q = self.ffn(h[1])
        h_i = h_i + self.dp_ffn(self.ls_ffn(ffn_i))
        h_q = h_q + self.dp_ffn(self.ls_ffn(ffn_q))

        return h_i, h_q


class DropPath(nn.Module):
    """Stochastic Depth per sample."""

    def __init__(self, prob: float = 0.0):
        super().__init__()
        self.prob = prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.prob == 0.0:
            return x
        keep = 1 - self.prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype) < keep
        return x * mask / keep


class RAPTORFMEncoder(nn.Module):
    """Full Radio-FM-style encoder: stack of encoder blocks + physical branch."""

    def __init__(
        self,
        d_model: int = 192,
        n_heads: int = 6,
        d_ff: int = 768,
        n_layers: int = 6,
        d_physical: int = 64,
        drop_path_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Progressive DropPath schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_max, n_layers)]
        self.blocks = nn.ModuleList([
            RAPTORFMEncoderBlock(d_model, n_heads, d_ff, drop_path=dpr[i])
            for i in range(n_layers)
        ])

        # Physical branch: lightweight conv path (NO saturating nonlinearities, NO bias — preserves scale)
        self.phys_conv = nn.Conv1d(d_physical, d_physical, kernel_size=3, padding=1, groups=d_physical, bias=False)
        self.phys_out = nn.Linear(d_physical, d_model, bias=False)  # project to d_model for fusion

    def forward(self, tok_out: dict):
        """
        tok_out: dict from RAPTORFMTokenizer.forward()
        Returns h_i, h_q (final CLS representations) and h_phys.
        """
        h_i = tok_out["i_masked"]
        h_q = tok_out["q_masked"]
        cos, sin = tok_out["cos"], tok_out["sin"]
        rope = RoPE(self.d_model)  # rope.apply is staticmethod-compatible

        # Encoder blocks
        for block in self.blocks:
            h_i, h_q = block(h_i, h_q, cos, sin, rope)

        # CLS representations (position 0)
        cls_i = h_i[:, 0]  # [B, D]
        cls_q = h_q[:, 0]  # [B, D]
        h_cls = (cls_i + cls_q) / 2.0  # average of two CLS tokens

        # Physical branch (linear path preserves scale)
        # Concatenate I and Q physical tokens: [B, 2*Lp, Dp]
        i_p = tok_out["i_p"]  # [B, Lp, Dp]
        q_p = tok_out["q_p"]
        phys = torch.cat([i_p, q_p], dim=1)  # [B, 2*Lp, Dp]
        # Transpose for conv1d: [B, Dp, 2*Lp]
        phys = phys.transpose(1, 2)
        phys = self.phys_conv(phys)  # depthwise conv, preserves scale
        phys = phys.transpose(1, 2)  # back to [B, 2*Lp, Dp]
        h_phys = phys.mean(dim=1)  # [B, Dp]
        h_phys = self.phys_out(h_phys)  # [B, D]

        return {
            "h_cls": h_cls,          # [B, D] normalized representation
            "h_phys": h_phys,        # [B, D] physical-scale representation
            "log_rms": tok_out["log_rms"],  # [B, 1]
            "h_i": h_i, "h_q": h_q,  # full sequence (for decoder)
        }

    def forward_full(self, tok_out: dict):
        """Forward returning full sequence too (for tests)."""
        out = self.forward(tok_out)
        return out
