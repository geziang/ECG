# -*- coding: utf-8 -*-
"""W9 后续: 逐记录预测底层数据归档(用户 10-20 指令: 仍有文件则直接归档)。

把 W4-W9/JOINT/ABC 各批 predictions/ 打 zip + 生成三域 canonical sample_order
(与加载器 sorted() 口径一致, split SHA 对冻结值核对) + archive_manifest.json,
git force-add 推送。不改动任何原始文件。
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runlog/preds_archive"
BATCHES = ["W4", "W5", "W6", "W7", "W8", "JOINT_20261013", "ABC_20261015", "W9"]
SPLIT_FROZEN = {"ptbxl": "5ca8de4ecd672a1e", "cpsc": "7808e41777518f4c",
                "chapman": "09783cad0542e5d1"}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_lines(ds):
    lines = []
    for split in ("train", "val", "test"):
        root = ROOT / "data" / ds / split
        lines.append(f"[{split}]")
        for cls in sorted(d for d in os.listdir(root) if (root / d).is_dir()):
            for fn in sorted(os.listdir(root / cls)):
                if fn.endswith(".npy"):
                    lines.append(f"{cls}/{fn}")
    return lines


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "purpose": "核心论文可复算底层数据: 逐记录预测+y_prob/y_true(+噪声offset)归档",
                "batches": {}}

    # 三域 canonical sample_order(现口径重导, split SHA 对冻结值)
    orders = {}
    for ds, want in SPLIT_FROZEN.items():
        lines = split_lines(ds)
        sha = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]
        ok = sha == want
        p = OUT / f"{ds}_sample_order.txt"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        orders[ds] = dict(sha=sha, frozen=want, ok=ok,
                          n=sum(1 for l in lines if not l.startswith("[")))
        print(f"sample_order {ds}: sha={sha} ok={ok} n={orders[ds]['n']}")
    manifest["sample_order"] = orders

    for b in BATCHES:
        src = ROOT / "runlog" / b / "predictions"
        if not src.exists():
            print(f"SKIP {b}: 无 predictions")
            continue
        zp = OUT / f"{b}_predictions.zip"
        n_files, n_npz = 0, 0
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in sorted(src.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(src).as_posix())
                    n_files += 1
                    if f.suffix == ".npy":
                        n_npz += 1
        size = zp.stat().st_size
        manifest["batches"][b] = dict(zip=zp.name, files=n_files, npy=n_npz,
                                      bytes=size, sha256=sha256_of(zp))
        print(f"{b}: {n_files} 文件(npy×{n_npz}) -> {zp.name} {size/1e6:.1f}MB")

    (OUT / "archive_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(v["bytes"] for v in manifest["batches"].values())
    print(f"TOTAL {total/1e6:.1f}MB; manifest -> archive_manifest.json")


if __name__ == "__main__":
    main()
