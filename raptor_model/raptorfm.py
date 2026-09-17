"""RAPTOR-FM: full model combining tokenizer, encoder, decoder, and fusion.

Faithful Radio-FM normalized branch + physical-scale branch + A/B/C fusion.
"""
import torch
import torch.nn as nn
from .raptorfm_tokenizer import RAPTORFMTokenizer
from .raptorfm_encoder import RAPTORFMEncoder


class RAPTORFMDecoder(nn.Module):
    """Lightweight decoder for masked reconstruction pretraining.
    Receives visible tokens + learnable mask tokens, reconstructs masked patches."""

    def __init__(self, d_model: int, d_decoder: int = 96, n_layers: int = 2, patch: int = 8):
        super().__init__()
        self.patch = patch
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Simple transformer decoder layers
        # Use nhead that divides d_decoder evenly
        nhead_dec = 4 if d_decoder % 4 == 0 else 2
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_decoder,
                nhead=nhead_dec,
                dim_feedforward=d_decoder * 4,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.input_proj = nn.Linear(d_model, d_decoder)
        self.output_proj = nn.Linear(d_decoder, d_model)

    def forward(self, h_i: torch.Tensor, h_q: torch.Tensor, mask_i: torch.Tensor, mask_q: torch.Tensor):
        """
        h_i, h_q: [B, L+1, D] encoder output (full sequence)
        mask_i, mask_q: [B, L+1] bool masks (True = masked)
        Returns reconstructed tokens for masked positions.
        """
        B, Lp1, D = h_i.shape

        # Replace masked positions with learnable mask token
        mt = self.mask_token.expand(B, Lp1, -1)
        dec_i = h_i.clone()
        dec_q = h_q.clone()
        dec_i[mask_i] = mt[mask_i]
        dec_q[mask_q] = mt[mask_q]

        # Project to decoder dim
        dec_i = self.input_proj(dec_i)
        dec_q = self.input_proj(dec_q)

        # Decode (self-attention over full sequence)
        for layer in self.layers:
            dec_i = layer(dec_i, dec_i)
            dec_q = layer(dec_q, dec_q)

        # Project back
        out_i = self.output_proj(dec_i)
        out_q = self.output_proj(dec_q)

        return out_i, out_q


class RAPTORFM(nn.Module):
    """Full RAPTOR-FM model.

    Modes:
        A: normalized-only  z = f(h_cls)
        B: physical-only    z = f(h_phys)
        C: fused            z = f(h_cls, h_phys)
    """

    def __init__(
        self,
        d_model: int = 192,
        n_heads: int = 6,
        d_ff: int = 768,
        n_layers: int = 6,
        d_physical: int = 64,
        d_decoder: int = 96,
        n_decoder_layers: int = 2,
        drop_path_max: float = 0.1,
        patch: int = 8,
        stride: int = 8,
        mask_ratio: float = 0.6,
        mode: str = "C",
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model

        self.tokenizer = RAPTORFMTokenizer(
            d_model=d_model,
            d_physical=d_physical,
            patch=patch,
            stride=stride,
            mask_ratio=mask_ratio,
            n_heads=n_heads,
        )
        self.encoder = RAPTORFMEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            d_physical=d_physical,
            drop_path_max=drop_path_max,
        )
        self.decoder = RAPTORFMDecoder(
            d_model=d_model,
            d_decoder=d_decoder,
            n_layers=n_decoder_layers,
            patch=patch,
        )

        # Fusion MLP
        if mode == "A":
            fusion_in = d_model
        elif mode == "B":
            fusion_in = d_model
        else:  # C
            fusion_in = d_model * 2 + 1  # h_cls + h_phys + log_rms
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Range head (downstream)
        self.range_head = nn.Linear(d_model, 1)

    def forward(self, iq: torch.Tensor, global_scale: torch.Tensor = None):
        """
        iq: [B, T, E=1, 2]
        Returns dict with all outputs.
        """
        # Tokenize
        tok_out = self.tokenizer(iq, global_scale)

        # Encode
        enc_out = self.encoder(tok_out)

        # Fusion
        if self.mode == "A":
            z = self.fusion(enc_out["h_cls"])
        elif self.mode == "B":
            z = self.fusion(enc_out["h_phys"])
        else:  # C
            z = self.fusion(torch.cat([
                enc_out["h_cls"],
                enc_out["h_phys"],
                enc_out["log_rms"],
            ], dim=-1))

        # Range prediction
        range_pred = self.range_head(z).squeeze(-1)

        # Decoder (for pretraining)
        dec_i, dec_q = self.decoder(
            enc_out["h_i"], enc_out["h_q"],
            tok_out["mask_i"], tok_out["mask_q"],
        )

        return {
            "z": z,                    # [B, D] fused representation
            "h_cls": enc_out["h_cls"],  # [B, D] normalized CLS
            "h_phys": enc_out["h_phys"],  # [B, D] physical
            "log_rms": enc_out["log_rms"],  # [B, 1]
            "range_pred": range_pred,  # [B]
            "dec_i": dec_i, "dec_q": dec_q,  # [B, L+1, D] reconstructions
            "mask_i": tok_out["mask_i"], "mask_q": tok_out["mask_q"],
            "i_t": tok_out["i_t"], "q_t": tok_out["q_t"],  # original tokens (targets)
        }

    def encode(self, iq: torch.Tensor, global_scale: torch.Tensor = None):
        """Encode without decoder (for downstream probing) -- unmasked, full sequence."""
        tok_out = self.tokenizer(iq, global_scale)
        tok_out["i_masked"] = tok_out["i_t"]
        tok_out["q_masked"] = tok_out["q_t"]
        enc_out = self.encoder(tok_out)
        if self.mode == "A":
            z = self.fusion(enc_out["h_cls"])
        elif self.mode == "B":
            z = self.fusion(enc_out["h_phys"])
        else:
            z = self.fusion(torch.cat([enc_out["h_cls"], enc_out["h_phys"], enc_out["log_rms"]], dim=-1))
        return z, enc_out