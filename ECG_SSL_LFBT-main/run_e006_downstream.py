"""Validation-first LP/FT runner for fixed E006 B0 and AR-B3 experiments."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_utils.cls_datasets import get_data_loaders
from data_utils.seed_utils import set_seed
from models.mbn import MultiBranchNet
from models.physiospatial import BASE_VARIANTS, base_variant_config
from utils.checkpoint import load_torch_checkpoint
from utils.metrics import runtime_metadata, save_evaluation, sha256_file, write_json


def build_parser():
    parser = argparse.ArgumentParser(description="E006 validation-first downstream evaluation")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("lp", "ft"), required=True)
    parser.add_argument("--base-variant", choices=BASE_VARIANTS, required=True)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--eval-split", choices=("val", "test"), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def evaluate(model, loader, device):
    model.eval()
    targets, probabilities = [], []
    with torch.no_grad():
        for data, target in tqdm(loader, desc="evaluate"):
            logits = model(data.to(device, non_blocking=True))
            targets.append(target.numpy())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def main():
    args = build_parser().parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        train_shuffle=True,
        train_ratio=args.fraction,
        seed=args.seed,
        return_metadata=True,
    )
    if len(class_names) != args.num_classes:
        raise ValueError(f"dataset contains {len(class_names)} classes, expected {args.num_classes}")
    base_config = base_variant_config(args.base_variant)
    source_checkpoint = load_torch_checkpoint(args.checkpoint, map_location="cpu")
    source_config = source_checkpoint.get("config", {})
    if source_checkpoint.get("base_variant") != args.base_variant:
        raise ValueError("checkpoint base variant does not match --base-variant")
    model = MultiBranchNet(
        args.num_classes,
        checkpoint=args.checkpoint,
        fusion=base_config["fusion"],
        map_location=device,
    ).to(device)
    if args.task == "lp":
        model.freeze_encoders()
        model.freeze_fusion()
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.001
    else:
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.0001
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    best_path = args.output_dir / "model_best.pth"
    best_val_loss = float("inf")
    config = {
        **vars(args),
        **runtime_metadata(),
        "class_names": class_names,
        "base_variant_definition": base_config,
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "profile": source_checkpoint.get("profile", source_config.get("profile")),
        "ablation": source_checkpoint.get("ablation", source_config.get("ablation")),
    }
    write_json(args.output_dir / "config.json", config)
    for epoch in range(args.epochs):
        model.train()
        if args.task == "lp":
            for encoder in model.encoder_group:
                encoder.eval()
            model.freeze_fusion()
        for data, target in tqdm(train_loader, desc=f"{args.task} epoch {epoch + 1}"):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(data.to(device, non_blocking=True)), target.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                val_loss += float(loss_fn(model(data.to(device, non_blocking=True)), target.to(device, non_blocking=True)))
        val_loss /= len(val_loader)
        print(f"epoch={epoch + 1} val_loss={val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
    model.load_state_dict(load_torch_checkpoint(best_path, map_location=device))
    evaluation_loader = val_loader if args.eval_split == "val" else test_loader
    y_true, y_prob = evaluate(model, evaluation_loader, device)
    metrics = save_evaluation(
        args.output_dir,
        y_true,
        y_prob,
        class_names,
        metadata={
            "task": args.task,
            "fraction": args.fraction,
            "eval_split": args.eval_split,
            "base_variant": args.base_variant,
            "fusion": base_config["fusion"],
            "seed": args.seed,
            "best_val_loss": best_val_loss,
            "model_sha256": sha256_file(best_path),
        },
    )
    print(f"{args.eval_split} AUROC={metrics['auroc']:.6f} AUPRC={metrics['auprc']:.6f}")


if __name__ == "__main__":
    main()
