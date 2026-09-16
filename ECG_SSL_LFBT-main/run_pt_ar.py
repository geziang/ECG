import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms

from data_utils.augmentations import RandomResizeCropTimeOut, ToTensor
from data_utils.data_folder import ECGDatasetFolder
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.seed_utils import set_seed
from data_utils.seed_utils_ar import make_generator, seed_worker
from models.ar_lfbt import LeadFusionBT
from utils.checkpoint import load_torch_checkpoint
from utils.metrics import json_ready, runtime_metadata, write_json


def build_parser():
    parser = argparse.ArgumentParser(description="LFBT/AR-LFBT pretraining")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--variant", default="B0")
    parser.add_argument("--num-leads", default=8, type=int)
    parser.add_argument("--input-length", default=2048, type=int)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.8, type=float)
    parser.add_argument("--lambd", default=0.0051, type=float)
    parser.add_argument("--projector", default="2048-2048-2048")
    parser.add_argument("--fusion", choices=["concat", "mean", "eca", "attention"], default="concat")
    parser.add_argument("--fusion-bt-weight", default=0.0, type=float)
    parser.add_argument("--attention-hidden-dim", default=32, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--lead-mask-prob", default=0.0, type=float)
    parser.add_argument("--min-masked-leads", default=1, type=int)
    parser.add_argument("--max-masked-leads", default=2, type=int)
    parser.add_argument("--print-freq", default=100, type=int)
    parser.add_argument("--save-every", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_dataset(dataset, num_leads, input_length):
    if len(dataset) < 2:
        raise ValueError("pretraining requires at least two samples")
    sample_path = Path(dataset.samples[0][0])
    sample = np.load(sample_path, mmap_mode="r")
    expected = (num_leads, input_length)
    if sample.shape != expected:
        raise ValueError(f"{sample_path} has shape {sample.shape}; expected {expected}")


def save_resume_state(path, epoch, model, optimizer):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    projector = tuple(int(value) for value in args.projector.split("-"))
    if not projector:
        raise ValueError("projector must contain at least one output dimension")

    transform = transforms.Compose([RandomResizeCropTimeOut(), ToTensor()])
    dataset = ECGDatasetFolder(
        args.data_dir,
        transform=MultiViewDataInjector([transform, transform]),
    )
    validate_dataset(dataset, args.num_leads, args.input_length)
    drop_last = len(dataset) % args.batch_size == 1
    if drop_last:
        print("Dropping a singleton final batch because BatchNorm requires B >= 2.")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed),
    )

    model = LeadFusionBT(
        num_leads=args.num_leads,
        projector=projector,
        gamma=args.gamma,
        lambd=args.lambd,
        fusion=args.fusion,
        fusion_bt_weight=args.fusion_bt_weight,
        attention_hidden_dim=args.attention_hidden_dim,
        dropout=args.dropout,
        lead_mask_probability=args.lead_mask_prob,
        min_masked_leads=args.min_masked_leads,
        max_masked_leads=args.max_masked_leads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    resume_path = args.checkpoint_dir / "pretrain_state.pth"
    start_epoch = 0
    if args.resume and resume_path.exists():
        state = load_torch_checkpoint(resume_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        print(f"Resuming from epoch {start_epoch}.")

    config = {
        **json_ready(vars(args)),
        **runtime_metadata(),
        "device": str(device),
        "dataset_size": len(dataset),
        "drop_last": drop_last,
    }
    write_json(args.checkpoint_dir / "config.json", config)
    history_path = args.checkpoint_dir / "history.jsonl"
    if start_epoch and history_path.exists():
        retained = []
        for line in history_path.read_text(encoding="utf-8").splitlines():
            try:
                if int(json.loads(line)["epoch"]) < start_epoch:
                    retained.append(line)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        history_path.write_text(
            ("\n".join(retained) + "\n") if retained else "",
            encoding="utf-8",
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = {key: 0.0 for key in ("loss", "loss_intra", "loss_inter", "loss_lfbt", "loss_fusion")}
        epoch_start = time.time()
        for step, ((view_a, view_b), _) in enumerate(loader):
            view_a = view_a.to(device, non_blocking=True)
            view_b = view_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(view_a, view_b)
            output["loss"].backward()
            optimizer.step()
            for key in totals:
                totals[key] += float(output[key].detach())
            if step % args.print_freq == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            **{key: round(float(output[key].detach()), 6) for key in totals},
                        }
                    )
                )
        epoch_metrics = {
            "epoch": epoch,
            "seconds": time.time() - epoch_start,
            **{key: value / len(loader) for key, value in totals.items()},
        }
        print(json.dumps(epoch_metrics))
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_resume_state(resume_path, epoch, model, optimizer)

    checkpoint_path = args.checkpoint_dir / "encoder_group.pth"
    checkpoint = {
        "format_version": 2,
        "variant": args.variant,
        "fusion": args.fusion,
        "backbone_state_dict_list": model.encoder_state_dict_list(),
        "fusion_state_dict": model.fusion.state_dict() if model.fusion is not None else None,
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    config["checkpoint_sha256"] = digest
    write_json(args.checkpoint_dir / "config.json", config)
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
