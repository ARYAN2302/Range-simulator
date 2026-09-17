"""RAPTOR-FM tokenizer: dual-branch input processing with RoPE, CI masking, CLS.

Faithful to Radio-FM (arXiv:2608.05793v1) §III-B/C:
- per-sequence RMS normalization for Branch N
- NO per-window normalization for Branch P (physical-scale branch)
- I/Q kept as two independent streams
- 1D conv patch embedding P=8/S=8 per channel
- learnable [CLS] per channel
- RoPE on Q/K using ORIGINAL patch positions (position-preserving under masking)
"""
import math
import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotary Positional Embedding applied to Q and K.
    
    Args:
        dim: head dimension (d_head = d_model // n_heads)
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        # dim must be even (head dimension)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))  # [seq_len, dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        return emb.cos(), emb.sin()

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # q, k: [B, H, L, D_h]
        # cos, sin: [L, D_h]
        cos = cos[None, None, :, :]  # [1, 1, L, D_h]
        sin = sin[None, None, :, :]
        return q * cos + self.rotate_half(q) * sin, k * cos + self.rotate_half(k) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class CIMaskGenerator:
    """Channel-Independent random masking: independent mask patterns for I and Q."""

    def __init__(self, mask_ratio: float = 0.6):
        self.mask_ratio = mask_ratio

    def __call__(self, batch_size: int, seq_len: int, device: torch.device):
        # Independent masks for I and Q
        mask_i = torch.rand(batch_size, seq_len, device=device) < self.mask_ratio
        mask_q = torch.rand(batch_size, seq_len, device=device) < self.mask_ratio
        return mask_i, mask_q  # each [B, L] bool


class RAPTORFMTokenizer(nn.Module):
    """Dual-branch tokenizer.

    Branch N (normalized): per-sequence RMS → I/Q split → conv patch → +CLS → +RoPE-ready positions
    Branch P (physical): NO per-window norm → lightweight pooling → conv → GRU

    Returns dict with all components needed by the encoder and decoder.
    """

    def __init__(
        self,
        d_model: int = 192,
        d_physical: int = 64,
        patch: int = 8,
        stride: int = 8,
        mask_ratio: float = 0.6,
        n_heads: int = 6,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_physical = d_physical
        self.patch = patch
        self.stride = stride
        self.mask_ratio = mask_ratio
        d_head = d_model // n_heads

        # Branch N: per-channel conv patch embedding (I and Q separate)
        self.proj_i = nn.Conv1d(1, d_model, kernel_size=patch, stride=stride)
        self.proj_q = nn.Conv1d(1, d_model, kernel_size=patch, stride=stride)

        # CLS tokens (one per channel)
        self.cls_i = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # RoPE (uses head dimension)
        self.rope = RoPE(d_head)

        # Normalization
        self.norm = RMSNorm(d_model)

        # Branch P: physical-scale branch (lightweight)
        # NO normalization that would remove amplitude/scale information
        self.pool = nn.AvgPool1d(kernel_size=64, stride=64)  # 8192 → 128
        self.phys_proj_i = nn.Conv1d(1, d_physical, kernel_size=3, padding=1, bias=False)
        self.phys_proj_q = nn.Conv1d(1, d_physical, kernel_size=3, padding=1, bias=False)
        # No RMSNorm here — we want to preserve scale

        # Mask generator
        self.mask_gen = CIMaskGenerator(mask_ratio)

    def forward(
        self,
        iq: torch.Tensor,
        global_scale: torch.Tensor = None,
    ) -> dict:
        """
        iq: [B, T, E=1, 2] float32 (I/Q)
        global_scale: [B] optional fixed scaling constant for Branch P

        Returns dict with all components needed by the encoder and decoder.
        """
        B, T, E, C = iq.shape
        assert C == 2, "last dim must be I/Q"
        device = iq.device

        # Split I/Q: each [B, T] → [B, 1, T] for Conv1d
        # iq shape: [B, T, E=1, C=2] → squeeze E dim → [B, T, 2]
        iq_squeezed = iq.squeeze(2)  # [B, T, 2]
        i = iq_squeezed[:, :, 0].unsqueeze(1)  # [B, 1, T]
        q = iq_squeezed[:, :, 1].unsqueeze(1)  # [B, 1, T]

        # ---- Branch N: normalized ----
        # per-sequence RMS
        sigma = torch.sqrt((iq.float() ** 2).mean(dim=(1, 2, 3), keepdim=True) + 1e-12)  # [B,1,1,1]
        sigma_flat = sigma.view(B)  # [B]
        i_n = iq_squeezed[:, :, 0] / sigma_flat.unsqueeze(-1)  # [B, T]
        q_n = iq_squeezed[:, :, 1] / sigma_flat.unsqueeze(-1)  # [B, T]
        i_n = i_n.unsqueeze(1)  # [B, 1, T]
        q_n = q_n.unsqueeze(1)  # [B, 1, T]

        # Patch embedding: [B, 1, T] → [B, D, L] → [B, L, D]
        i_t = self.proj_i(i_n).transpose(1, 2)  # [B, L, D]
        q_t = self.proj_q(q_n).transpose(1, 2)  # [B, L, D]
        L = i_t.shape[1]

        # Prepend CLS: [B, L+1, D]
        cls_i = self.cls_i.expand(B, -1, -1)
        cls_q = self.cls_q.expand(B, -1, -1)
        i_t = torch.cat([cls_i, i_t], dim=1)  # [B, L+1, D]
        q_t = torch.cat([cls_q, q_t], dim=1)  # [B, L+1, D]

        # Normalize
        i_t = self.norm(i_t)
        q_t = self.norm(q_t)

        # RoPE (original positions, including CLS at position 0)
        cos, sin = self.rope(L + 1, device)

        # CI masking (on patch positions only, not CLS)
        mask_i, mask_q = self.mask_gen(B, L, device)
        # Prepend False for CLS (never masked)
        mask_i = torch.cat([torch.zeros(B, 1, device=device, dtype=torch.bool), mask_i], dim=1)
        mask_q = torch.cat([torch.zeros(B, 1, device=device, dtype=torch.bool), mask_q], dim=1)

        # Apply mask (zero out masked positions)
        i_masked = i_t.clone()
        q_masked = q_t.clone()
        i_masked[mask_i] = 0.0
        q_masked[mask_q] = 0.0

        # ---- Branch P: physical-scale ----
        # NO per-window RMS; optional global scale for numerical stability
        if global_scale is not None:
            scale = global_scale.view(B, 1, 1)
        else:
            scale = 1.0
        i_p = (iq_squeezed[:, :, 0].unsqueeze(1) / scale)  # [B, 1, T]
        q_p = (iq_squeezed[:, :, 1].unsqueeze(1) / scale)  # [B, 1, T]

        # Lightweight pooling + conv
        i_p = self.pool(i_p)  # [B, 1, Lp]
        q_p = self.pool(q_p)
        i_p = self.phys_proj_i(i_p).transpose(1, 2)  # [B, Lp, Dp]
        q_p = self.phys_proj_q(q_p).transpose(1, 2)
        # No normalization — preserve physical scale

        # log-RMS scalar physical token
        log_rms = torch.log(sigma.squeeze(-1).squeeze(-1).squeeze(-1) + 1e-12).unsqueeze(-1)  # [B, 1]

        return {
            "i_t": i_t, "q_t": q_t,           # normalized tokens [B, L+1, D]
            "i_masked": i_masked, "q_masked": q_masked,  # masked versions
            "mask_i": mask_i, "mask_q": mask_q,  # bool masks [B, L+1]
            "cos": cos, "sin": sin,            # RoPE
            "i_p": i_p, "q_p": q_p,           # physical tokens [B, Lp, Dp]
            "log_rms": log_rms,                # [B, 1]
            "cls_pos": 0,                      # CLS at position 0
        }