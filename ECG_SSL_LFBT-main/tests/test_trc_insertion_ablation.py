#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded unit checks for the opt-in TRC insertion-position probe.

Run from ``ECG_SSL_LFBT-main`` with ``python tests/test_trc_insertion_ablation.py``.
The test deliberately never starts a pretraining job.  On a documentation-only
machine without PyTorch it exits cleanly with a SKIP message.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import torch
except Exception as exc:  # pragma: no cover - host-dependent
    print(f"SKIP TRC insertion smoke: PyTorch unavailable ({exc})")
    raise SystemExit(0)

from tools.trc_insertion_ablation import (  # noqa: E402
    MultiLeadTRCProbe,
    POSITIONS,
    benchmark_one,
    parameter_summary,
)


def main() -> None:
    seed = 0
    torch.manual_seed(seed)
    models = {position: MultiLeadTRCProbe(position, num_leads=2) for position in POSITIONS}
    x = torch.randn(2, 2, 256)
    outputs = {position: model(x) for position, model in models.items()}

    assert all(tuple(value.shape) == (2, 2, 64) for value in outputs.values())
    base_total, base_trc, _ = parameter_summary(models["off"])
    pre_total, pre_trc, pre_share = parameter_summary(models["pre_gap"])
    post_total, post_trc, post_share = parameter_summary(models["post_gap"])
    expected_delta = 2 * 2 * 64  # 2 leads × (gamma,beta) × 64 channels
    assert base_trc == 0
    assert pre_trc == expected_delta and post_trc == expected_delta
    assert pre_total - base_total == expected_delta
    assert post_total - base_total == expected_delta
    assert pre_share > 0.0 and post_share > 0.0

    # Zero-initialized GRN must preserve the B0 forward exactly at construction.
    for position in ("pre_gap", "post_gap"):
        models[position].load_state_dict(models["off"].state_dict(), strict=False)
        # The preceding line only shares compatible convolution keys; compare
        # each model against a freshly built B0 with the same seed below.
    torch.manual_seed(seed)
    reference = MultiLeadTRCProbe("off", num_leads=2)
    for position in ("pre_gap", "post_gap"):
        torch.manual_seed(seed)
        probe = MultiLeadTRCProbe(position, num_leads=2)
        probe.load_state_dict(reference.state_dict(), strict=False)
        assert torch.equal(reference(x), probe(x)), f"{position} is not zero-init equivalent"

    row = benchmark_one(
        "post_gap",
        seed=0,
        device_name="cpu",
        num_leads=1,
        batch_size=1,
        signal_length=64,
        warmup_steps=0,
        measure_steps=1,
    )
    assert row["status"] == "ok"
    assert row["params_trc"] == 128
    assert row["checkpoint_load_status"] == "none"
    print("PASS TRC insertion probe: shape, zero-init, parameter delta, bounded CPU smoke")


if __name__ == "__main__":
    main()

