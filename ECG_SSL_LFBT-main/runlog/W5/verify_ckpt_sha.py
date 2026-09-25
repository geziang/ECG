# -*- coding: utf-8 -*-
"""W5 A-2 前置: checkpoint SHA256 反查核验 (freeze manifest + W4 baseline_results 对照)。

预期值来源:
  - b0 s0  : runlog/W2/freeze_manifest_v2.json "b0_anchor(W1)" (predreplay CK_OF: checkpoint/ptxl_gamma08)
  - b0 s2/s4, c1/c2 x3 : freeze_manifest_v2.json "b0_seed2/4","c1_seed*","c2_seed*"
  - simclr x3, clocs x3 : runlog/W4/baseline_results.csv checkpoint_sha256 列
任一不符 -> exit 1 (任务书红线: 中间态对不上即停)。
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXPECT = {
    ("b0", 0): ("checkpoint/ptxl_gamma08/encoder_group.pth",
                "026033d532d4e42a1b53ca9c4d2e95f9c09b2daaaab5c9041c9d474a1356cdad",
                "freeze_manifest_v2:b0_anchor(W1)"),
    ("b0", 2): ("checkpoint/confirm/b0_seed2/encoder_group.pth",
                "f39059a4430c2e7aa8963ca3c75b608120452f517622862fb3a73a7dc95d26a4",
                "freeze_manifest_v2:b0_seed2"),
    ("b0", 4): ("checkpoint/confirm/b0_seed4/encoder_group.pth",
                "ba4f2d3c99aff6b5889dae4ebdd1103b8949e2d1f8fcadd28d0a5df8b91c0daa",
                "freeze_manifest_v2:b0_seed4"),
    ("c1", 0): ("checkpoint/confirm/c1_seed0/encoder_group.pth",
                "de0c28eedb30841492c36644ad1d06e11aab7bb02bbb5b42ff5d18524a5bc177",
                "freeze_manifest_v2:c1_seed0"),
    ("c1", 2): ("checkpoint/confirm/c1_seed2/encoder_group.pth",
                "26ebf0e26e5abee04f4064182d79e449e3c849693979162577171717cb2b7787",
                "freeze_manifest_v2:c1_seed2"),
    ("c1", 4): ("checkpoint/confirm/c1_seed4/encoder_group.pth",
                "e599dbcdb69372186c3b49b0d0ced3389e2c909a458d86a8413a58524e31715e",
                "freeze_manifest_v2:c1_seed4"),
    ("c2", 0): ("checkpoint/confirm/c2_seed0/encoder_group.pth",
                "af242a89c340969fd57e70349e4b88e4bf6725c5586f5c12d62a96a199ed9524",
                "freeze_manifest_v2:c2_seed0"),
    ("c2", 2): ("checkpoint/confirm/c2_seed2/encoder_group.pth",
                "d3f8979f70cdce3da7eefa320f72d98ba5a975a5ea4141605fbf52ce34de5666",
                "freeze_manifest_v2:c2_seed2"),
    ("c2", 4): ("checkpoint/confirm/c2_seed4/encoder_group.pth",
                "5979f230172000c51b9de93fc6d96d675c844841610f0b71cbb0c8e0147c982d",
                "freeze_manifest_v2:c2_seed4"),
    ("simclr", 0): ("checkpoint/confirm/simclr_seed0/encoder_group.pth",
                    "cc28ed3a955490014cb582c9fb75193140b9ca5d083e3f783963b76c80d2f103",
                    "W4/baseline_results.csv:simclr s0"),
    ("simclr", 2): ("checkpoint/confirm/simclr_seed2/encoder_group.pth",
                    "e4b431ba5c601bd52de3ad0e9b70d003d6840b5b8ce81409b50cae2b44d8b677",
                    "W4/baseline_results.csv:simclr s2"),
    ("simclr", 4): ("checkpoint/confirm/simclr_seed4/encoder_group.pth",
                    "5958c81aeef97f9a35144740bec47216f5b3cd8e023737f19dbdca85c328fc30",
                    "W4/baseline_results.csv:simclr s4"),
    ("clocs", 0): ("checkpoint/confirm/clocs_seed0/encoder_group.pth",
                   "c1f139d98059812aeee6453c5c7ec9cb2e99d57f338178833cd6e0cd159574d5",
                   "W4/baseline_results.csv:clocs s0"),
    ("clocs", 2): ("checkpoint/confirm/clocs_seed2/encoder_group.pth",
                   "69bc974a433ae50222178a4d084e07ed175254a3100034c7ad06451d2c4f11c1",
                   "W4/baseline_results.csv:clocs s2"),
    ("clocs", 4): ("checkpoint/confirm/clocs_seed4/encoder_group.pth",
                   "914ed37ad35e05d70cfb00c217f5f622dfd6eef0441507ba02a5da968f13e995",
                   "W4/baseline_results.csv:clocs s4"),
}


def main():
    rows, bad = [], 0
    for (kind, seed), (rel, exp_sha, src) in sorted(EXPECT.items()):
        p = ROOT / rel
        if not p.exists():
            print(f"[MISS] {kind} s{seed}: {rel} 不存在")
            bad += 1
            rows.append({"kind": kind, "seed": seed, "path": rel, "sha256": None,
                         "expect": exp_sha, "match": False, "src": src})
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        ok = h == exp_sha
        if not ok:
            bad += 1
        print(f"[{'OK' if ok else 'MISMATCH'}] {kind} s{seed}: {h[:16]}.. vs {exp_sha[:16]}..  ({src})")
        rows.append({"kind": kind, "seed": seed, "path": rel, "sha256": h,
                     "expect": exp_sha, "match": ok, "src": src})
    out = ROOT / "runlog/W5/ckpt_sha_check.json"
    out.write_text(json.dumps({"checked": len(rows), "mismatch": bad, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(rows)-bad}/{len(rows)} 一致; 明细 {out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
