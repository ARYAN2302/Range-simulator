#!/usr/bin/env python3
"""L1: standalone RSSI feature -- no model, no trainable parameters, no dependency on L2.

This exists as its own module because L1 was previously computed by instantiating the full
RAPTORFM model and reading `log_rms` out of its tokenizer's forward pass -- backwards, since
L1 is supposed to be a cheap, dependency-free measurement you can compute instantly on raw IQ,
independent of whether the (large, GPU-bound) L2 model is even loaded. Same pattern as the
production two-antenna DOA repo's own RSSI-equivalent (`std = x_raw.std(dim=(1,2,3))`,
computed as a bare tensor op before their phase-only normalization strips it out) -- just
extracted here into its own function instead of buried inside a neural network's forward pass.
"""
import numpy as np


def compute_log_rms(iq: np.ndarray) -> np.ndarray:
    """iq: [..., T] complex, or [..., T, 2] real/imag. Returns log(RMS amplitude), same
    leading shape minus the time axis. Pure arithmetic -- no model, no parameters.

    Matches RAPTORFMTokenizer's own sigma exactly: it averages I and Q as separate samples
    in one pool of 2T real values (`iq.float()**2).mean(dim=(1,2,3))` over a [T,1,2] tensor),
    which is mean_t(I^2+Q^2)/2, not mean_t(I^2+Q^2) -- a sqrt(2) difference that matters for
    exact reproducibility even though it's just a constant any calibration offset absorbs.
    """
    if np.iscomplexobj(iq):
        sum_sq = iq.real ** 2 + iq.imag ** 2  # [..., T]
    else:
        sum_sq = (iq ** 2).sum(axis=-1)  # [..., T, 2] -> [..., T]
    sigma = np.sqrt(sum_sq.mean(axis=-1) / 2 + 1e-12)  # mean over (T, I/Q) combined -> [...]
    return np.log(sigma + 1e-12)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    iq_complex = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)
    iq_real = np.stack([iq_complex.real, iq_complex.imag], axis=-1)
    r1 = compute_log_rms(iq_complex)
    r2 = compute_log_rms(iq_real)
    print("log_rms (complex input):", r1)
    print("log_rms (real/imag input):", r2)
    assert np.isclose(r1, r2), "both input conventions must agree"
    print("OK -- L1 is a standalone function, no model required.")
