#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure TRC insertion-position overhead without touching the W2 pipeline.

This is an opt-in smoke/benchmark tool.  It builds the existing VGG16
backbone, inserts the already-frozen ``GRN1D`` implementation either before
the global average pool (``pre_gap``; the current C2 placement) or after it
(``post_gap``), and measures parameter count, forward time, one training-step
time, and peak CUDA memory on synthetic input.

No pretraining or downstream evaluation is started by this module.  A
checkpoint may be supplied only to verify/load frozen convolution weights for
the overhead measurement.  Post-GAP is a new architecture and must not be
treated as a trained result merely because it inherited baseline weights.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # Keep ``--help`` useful even on a CPU-only analysis machine.
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - exercised only without torch
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


POSITIONS: Tuple[str, ...] = ("off", "pre_gap", "post_gap")
CSV_FIELDS: Tuple[str, ...] = (
    "timestamp_utc",
    "mode",
    "seed",
    "device",
    "cuda_name",
    "torch_version",
    "num_leads",
    "alpha",
    "batch_size",
    "signal_length",
    "warmup_steps",
    "measure_steps",
    "params_total",
    "params_trc",
    "trc_share_pct",
    "forward_ms",
    "train_step_ms",
    "throughput_samples_per_s",
    "peak_memory_mb",
    "output_checksum",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_config_sha256",
    "checkpoint_load_status",
    "checkpoint_arch_hint",
    "git_sha",
    "status",
    "error",
)


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - depends on host environment
        raise RuntimeError(
            "PyTorch is required for the TRC smoke benchmark; import failed: "
            f"{_TORCH_IMPORT_ERROR}"
        )


def sha256_file(path: Optional[Path]) -> str:
    """Return a full SHA256, or an empty string for an absent path."""

    if path is None or not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_sha(root: Path = ROOT) -> str:
    """Read the exact code revision without making git state a dependency."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _checkpoint_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    if path.is_dir():
        path = path / "encoder_group.pth"
    return path


def _is_trc_key(key: str) -> bool:
    # State keys emitted by VGG16 are model.<index>.gamma/beta.  Keep this
    # intentionally narrow: unrelated missing keys must still fail loudly.
    parts = key.split(".")
    return len(parts) == 3 and parts[0] == "model" and parts[2] in {"gamma", "beta"}


def _state_dict_from_item(item: object) -> Mapping[str, object]:
    if hasattr(item, "state_dict"):
        return item.state_dict()  # type: ignore[no-any-return]
    if isinstance(item, Mapping):
        return item
    raise TypeError(f"unsupported encoder checkpoint item: {type(item)!r}")


def _checkpoint_states(checkpoint: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    if "backbone_state_dict_list" in checkpoint:
        raw = checkpoint["backbone_state_dict_list"]
    elif "backbone_state_dict" in checkpoint:
        raw = checkpoint["backbone_state_dict"]
    else:
        raise ValueError("checkpoint has neither backbone_state_dict_list nor backbone_state_dict")
    if not isinstance(raw, Sequence):
        raise TypeError("checkpoint encoder list is not a sequence")
    return [_state_dict_from_item(item) for item in raw]


def checkpoint_arch_hint(path: Optional[Path]) -> str:
    """Return a compact, human-readable architecture hint from config.json."""

    if path is None:
        return ""
    cfg_path = path.parent / "config.json"
    if not cfg_path.exists():
        return "config_missing"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return "config_unreadable"
    args = cfg.get("args", cfg)
    if not isinstance(args, Mapping):
        return "config_no_args"
    values = []
    for key in ("trc", "blur_pool", "pool_power", "num_leads"):
        if key in args:
            values.append(f"{key}={args[key]}")
    return ",".join(values) if values else "config_no_arch_fields"


def _build_encoder(position: str, alpha: float = 0.125):
    """Build one encoder while leaving ``models/vgg_1d.py`` untouched."""

    _require_torch()
    if position not in POSITIONS:
        raise ValueError(f"unknown TRC position: {position}")
    from models.vgg_1d import GRN1D, VGG16

    # Start from the B0 architecture.  This preserves all existing block and
    # pooling state keys; only the opt-in GRN module is added below.
    encoder = VGG16(ch_in=1, alpha=alpha, trc=0)
    encoder.fc = nn.Identity()
    if position == "off":
        return encoder

    blocks = list(encoder.model.children())
    if not blocks or not isinstance(blocks[-1], nn.AdaptiveAvgPool1d):
        raise AssertionError("unexpected VGG16 tail; TRC probe expects an AdaptiveAvgPool1d")
    trc = GRN1D(encoder.output_dim)
    if position == "pre_gap":
        blocks.insert(len(blocks) - 1, trc)
    else:
        blocks.append(trc)
    encoder.model = nn.Sequential(*blocks)
    # Metadata is deliberately non-persistent: it does not alter checkpoint
    # keys and cannot accidentally affect the production code path.
    encoder._trc_probe_position = position
    return encoder


class MultiLeadTRCProbe(nn.Module):
    """Eight independent LFBT lead encoders for overhead-only measurement."""

    def __init__(self, position: str, num_leads: int = 8, alpha: float = 0.125):
        super().__init__()
        if num_leads < 1:
            raise ValueError("num_leads must be positive")
        self.position = position
        self.num_leads = int(num_leads)
        self.alpha = float(alpha)
        self.encoders = nn.ModuleList(
            [_build_encoder(position, alpha=self.alpha) for _ in range(self.num_leads)]
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.num_leads:
            raise ValueError(
                f"expected [B,{self.num_leads},T], received {tuple(x.shape)}"
            )
        features = [
            encoder(x[:, index : index + 1, :])
            for index, encoder in enumerate(self.encoders)
        ]
        return torch.stack(features, dim=1)


def _trc_parameters(model: nn.Module) -> List[object]:
    from models.vgg_1d import GRN1D

    params = []
    for module in model.modules():
        if isinstance(module, GRN1D):
            params.extend(list(module.parameters()))
    return params


def parameter_summary(model: nn.Module) -> Tuple[int, int, float]:
    total = sum(int(param.numel()) for param in model.parameters())
    trc = sum(int(param.numel()) for param in _trc_parameters(model))
    share = 100.0 * trc / total if total else 0.0
    return total, trc, share


def _load_frozen_checkpoint(model: nn.Module, path: Optional[Path]) -> Tuple[str, str]:
    """Load encoder weights and return ``(status, arch_hint)``.

    A baseline checkpoint may seed either insertion-position smoke model.  The
    only tolerated mismatch is the optional GRN ``gamma/beta`` pair; all
    convolution/normalization mismatches are errors.  This makes a typo in a
    checkpoint path visible instead of silently benchmarking random weights.
    """

    _require_torch()
    path = _checkpoint_path(path)
    if path is None:
        return "none", ""
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint root must be a mapping, got {type(checkpoint)!r}")
    states = _checkpoint_states(checkpoint)
    encoders = list(getattr(model, "encoders", []))
    if len(states) != len(encoders):
        raise ValueError(f"checkpoint has {len(states)} leads, model expects {len(encoders)}")

    tolerated = []
    for index, (encoder, state) in enumerate(zip(encoders, states)):
        target = encoder.state_dict()
        normalized = {}
        for key, value in state.items():
            # Older module-list checkpoints may expose ``module`` or ``fc``
            # entries; only encoder ``model.*`` tensors belong here.
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("fc."):
                continue
            if key in target:
                normalized[key] = value
            elif _is_trc_key(key):
                tolerated.append(f"lead{index}:{key}")
            else:
                raise ValueError(f"lead {index} unexpected checkpoint key: {key}")
        missing, unexpected = encoder.load_state_dict(normalized, strict=False)
        bad_missing = [key for key in missing if not _is_trc_key(key)]
        bad_unexpected = [key for key in unexpected if not _is_trc_key(key)]
        if bad_missing or bad_unexpected:
            raise ValueError(
                f"lead {index} checkpoint mismatch: missing={bad_missing}, "
                f"unexpected={bad_unexpected}"
            )
        tolerated.extend(f"lead{index}:{key}" for key in missing if _is_trc_key(key))
    status = "loaded" if not tolerated else "loaded_with_trc_position_mismatch"
    return status, checkpoint_arch_hint(path)


def _sync(device: object) -> None:
    if torch is not None and getattr(device, "type", str(device)) == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_forward(model, x, steps: int, device) -> float:
    _sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            output = model(x)
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / steps


def _elapsed_train(model, x, steps: int, device) -> float:
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    _sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        # A local scalar objective is enough to exercise the same autograd path
        # for all positions; this is intentionally not a pretraining loss.
        loss = output.square().mean()
        loss.backward()
        optimizer.step()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / steps


def benchmark_one(
    position: str,
    *,
    seed: int = 0,
    device_name: str = "auto",
    num_leads: int = 8,
    alpha: float = 0.125,
    batch_size: int = 2,
    signal_length: int = 2048,
    warmup_steps: int = 1,
    measure_steps: int = 2,
    checkpoint: Optional[Path] = None,
) -> Dict[str, object]:
    """Run a bounded synthetic smoke benchmark and return one CSV row."""

    _require_torch()
    if position not in POSITIONS:
        raise ValueError(position)
    if batch_size < 1 or signal_length < 32:
        raise ValueError("batch_size must be positive and signal_length >= 32")
    if warmup_steps < 0 or measure_steps < 1:
        raise ValueError("warmup_steps >= 0 and measure_steps >= 1 are required")

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = MultiLeadTRCProbe(position, num_leads=num_leads, alpha=alpha).to(device)
    checkpoint_path = _checkpoint_path(checkpoint)
    load_status, arch_hint = _load_frozen_checkpoint(model, checkpoint_path)
    model.train()
    x = torch.randn(batch_size, num_leads, signal_length, device=device)

    # Keep all modes on the same synthetic input and initial random seed.  The
    # GRN parameters are zero-initialized, so pre/post smoke starts at B0.
    for _ in range(warmup_steps):
        _ = _elapsed_forward(model, x, 1, device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    forward_ms = _elapsed_forward(model, x, measure_steps, device)
    train_step_ms = _elapsed_train(model, x, measure_steps, device)
    peak_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        if device.type == "cuda"
        else -1.0
    )
    with torch.no_grad():
        output = model(x)
    checksum = float(output.detach().mean().cpu().item())
    params_total, params_trc, trc_share = parameter_summary(model)
    cuda_name = torch.cuda.get_device_name(device) if device.type == "cuda" else ""
    elapsed_s = train_step_ms / 1000.0 * measure_steps
    throughput = (batch_size * measure_steps / elapsed_s) if elapsed_s > 0 else math.nan

    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "mode": position,
        "seed": int(seed),
        "device": str(device),
        "cuda_name": cuda_name,
        "torch_version": str(torch.__version__),
        "num_leads": int(num_leads),
        "alpha": float(alpha),
        "batch_size": int(batch_size),
        "signal_length": int(signal_length),
        "warmup_steps": int(warmup_steps),
        "measure_steps": int(measure_steps),
        "params_total": int(params_total),
        "params_trc": int(params_trc),
        "trc_share_pct": round(float(trc_share), 6),
        "forward_ms": round(float(forward_ms), 4),
        "train_step_ms": round(float(train_step_ms), 4),
        "throughput_samples_per_s": round(float(throughput), 4),
        "peak_memory_mb": round(float(peak_mb), 4),
        "output_checksum": f"{checksum:.10g}",
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_config_sha256": sha256_file(
            checkpoint_path.parent / "config.json" if checkpoint_path else None
        ),
        "checkpoint_load_status": load_status,
        "checkpoint_arch_hint": arch_hint,
        "git_sha": current_git_sha(),
        "status": "ok",
        "error": "",
    }


def _write_rows(rows: Sequence[Mapping[str, object]], output_csv: Optional[Path]) -> None:
    target = sys.stdout
    close_target = False
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        target = output_csv.open("w", newline="", encoding="utf-8")
        close_target = True
    try:
        writer = csv.DictWriter(target, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    finally:
        if close_target:
            target.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["all", *POSITIONS],
        default="all",
        help="TRC placement to benchmark; all runs off/pre_gap/post_gap.",
    )
    parser.add_argument("--seed", type=int, default=0, help="smoke seed (default: 0)")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--num-leads", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.125)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--signal-length", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, default=None, help="optional frozen encoder_group.pth")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="bounded seed-0 smoke: batch=2, warmup=1, measure=2 (overrides those values)",
    )
    parser.add_argument(
        "--allow-long",
        action="store_true",
        help="allow more than 20 benchmark steps; never needed for the smoke task",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.smoke:
        args.seed = 0
        args.batch_size = 2
        args.warmup_steps = 1
        args.measure_steps = 2
    if args.measure_steps > 20 and not args.allow_long:
        parser.error("measure-steps > 20 requires --allow-long; use --smoke for the approved probe")
    positions = list(POSITIONS) if args.mode == "all" else [args.mode]
    rows = []
    for position in positions:
        try:
            rows.append(
                benchmark_one(
                    position,
                    seed=args.seed,
                    device_name=args.device,
                    num_leads=args.num_leads,
                    alpha=args.alpha,
                    batch_size=args.batch_size,
                    signal_length=args.signal_length,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                    checkpoint=args.checkpoint,
                )
            )
        except Exception as exc:
            # Emit a schema-complete failed row, so a failed mode is still
            # auditable and cannot be mistaken for a missing experiment.
            rows.append(
                {
                    "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                    "mode": position,
                    "seed": args.seed,
                    "device": args.device,
                    "num_leads": args.num_leads,
                    "alpha": args.alpha,
                    "batch_size": args.batch_size,
                    "signal_length": args.signal_length,
                    "warmup_steps": args.warmup_steps,
                    "measure_steps": args.measure_steps,
                    "checkpoint_path": str(_checkpoint_path(args.checkpoint) or ""),
                    "checkpoint_sha256": sha256_file(_checkpoint_path(args.checkpoint)),
                    "checkpoint_config_sha256": sha256_file(
                        (_checkpoint_path(args.checkpoint).parent / "config.json")
                        if _checkpoint_path(args.checkpoint)
                        else None
                    ),
                    "git_sha": current_git_sha(),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    _write_rows(rows, args.output_csv)
    return 0 if all(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

