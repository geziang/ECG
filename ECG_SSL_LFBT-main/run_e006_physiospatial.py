"""Train one pre-registered E006 physiological-spatial LFBT hypothesis."""

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
from models.physiospatial import (
    ABLATIONS,
    BASE_VARIANTS,
    RESEARCH_PROFILES,
    PhysioSpatialLFBT,
    base_variant_config,
)
from utils.metrics import json_ready, runtime_metadata, write_json


def parser():
    value = argparse.ArgumentParser(description="E006 structured physiological-spatial LFBT")
    value.add_argument("--data-dir", type=Path, required=True)
    value.add_argument("--checkpoint-dir", type=Path, required=True)
    value.add_argument("--profile", choices=tuple(RESEARCH_PROFILES), required=True)
    value.add_argument("--base-variant", choices=BASE_VARIANTS, required=True)
    value.add_argument("--ablation", choices=ABLATIONS, default="full")
    value.add_argument("--num-leads", default=8, type=int)
    value.add_argument("--input-length", default=2048, type=int)
    value.add_argument("--epochs", default=200, type=int)
    value.add_argument("--batch-size", default=128, type=int)
    value.add_argument("--workers", default=6, type=int)
    value.add_argument("--learning-rate", default=0.001, type=float)
    value.add_argument("--gamma", default=0.8, type=float)
    value.add_argument("--lambd", default=0.0051, type=float)
    value.add_argument("--projector", default="2048-2048-2048")
    value.add_argument("--physiology-weight", default=0.05, type=float)
    value.add_argument("--spatial-weight", default=0.05, type=float)
    value.add_argument("--print-freq", default=100, type=int)
    value.add_argument("--seed", default=0, type=int)
    return value


def validate_dataset(dataset, leads, length):
    if len(dataset) < 2:
        raise ValueError("pretraining needs at least two ECG records")
    sample_path = Path(dataset.samples[0][0])
    if np.load(sample_path, mmap_mode="r").shape != (leads, length):
        raise ValueError(f"{sample_path} does not satisfy the [{leads}, {length}] LFBT input contract")


def main():
    args = parser().parse_args()
    set_seed(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augmentation = transforms.Compose([RandomResizeCropTimeOut(), ToTensor()])
    dataset = ECGDatasetFolder(args.data_dir, transform=MultiViewDataInjector([augmentation, augmentation]))
    validate_dataset(dataset, args.num_leads, args.input_length)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) % args.batch_size == 1,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed),
    )
    model = PhysioSpatialLFBT(
        profile=args.profile,
        base_variant=args.base_variant,
        ablation=args.ablation,
        num_leads=args.num_leads,
        projector=tuple(int(item) for item in args.projector.split("-")),
        gamma=args.gamma,
        lambd=args.lambd,
        physiology_weight=args.physiology_weight,
        spatial_weight=args.spatial_weight,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    configuration = {
        **json_ready(vars(args)),
        **runtime_metadata(),
        "device": str(device),
        "dataset_size": len(dataset),
        "profile_definition": RESEARCH_PROFILES[args.profile],
        "base_variant_definition": base_variant_config(args.base_variant),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    write_json(args.checkpoint_dir / "config.json", configuration)
    history_path = args.checkpoint_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)
    metrics = (
        "loss",
        "loss_intra",
        "loss_inter",
        "loss_lfbt",
        "loss_fusion",
        "loss_base",
        "loss_physiology",
        "loss_spatial",
    )
    for epoch in range(args.epochs):
        model.train()
        total = {key: 0.0 for key in metrics}
        started = time.time()
        for step, ((view_a, view_b), _) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            result = model(view_a.to(device, non_blocking=True), view_b.to(device, non_blocking=True))
            result["loss"].backward()
            optimizer.step()
            for key in metrics:
                total[key] += float(result[key].detach())
            if step % args.print_freq == 0:
                print(json.dumps({"epoch": epoch, "step": step, **{key: float(result[key].detach()) for key in metrics}}))
        record = {"epoch": epoch, "seconds": time.time() - started, **{key: value / len(loader) for key, value in total.items()}}
        print(json.dumps(record))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    checkpoint = args.checkpoint_dir / "encoder_group.pth"
    torch.save({
        "format_version": 4,
        "variant": "E006",
        "profile": args.profile,
        "base_variant": args.base_variant,
        "ablation": args.ablation,
        "fusion": model.fusion_name,
        "fusion_bt_weight": model.fusion_bt_weight,
        "lead_mask_probability": model.lead_mask.probability,
        "backbone_state_dict_list": model.encoder_state_dict_list(),
        "fusion_state_dict": model.fusion.state_dict() if model.fusion is not None else None,
        "config": configuration,
    }, checkpoint)
    configuration["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_json(args.checkpoint_dir / "config.json", configuration)
    print(f"Checkpoint saved: {checkpoint}")


if __name__ == "__main__":
    main()
