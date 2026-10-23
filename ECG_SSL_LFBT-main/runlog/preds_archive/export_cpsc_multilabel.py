# -*- coding: utf-8 -*-
"""CPSC 原生多标签真值导出(2026-10-20, 用户指令: 找回多标签底层数据)。

源: F:\\数据集\\...challenge-2020-1.0.2\\training\\cpsc_2018\\g*\\*.hea 的完整 # Dx: 行。
对齐: data/cpsc canonical 顺序(train/val/test × 类目录序 × 文件名序, split SHA 已冻结)。
校验: ①6877 条一一对应 ②priority 主类==目录放置 100% 一致(与 prepare_cpsc_chapman 同口径)。
产出: runlog/preds_archive/cpsc_multilabel/{cpsc_multilabel.csv, cpsc_multilabel.npz, summary.json}
"""
import csv
import io
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW_GLOB = r"F:\数据集\classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2\training\cpsc_2018\g*\*.hea"
OUTD = ROOT / "runlog/preds_archive/cpsc_multilabel"
CPSC_CLASSES = ["NSR", "AF", "IAVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
CPSC_MAP = {
    "426783006": "NSR", "164889003": "AF", "270492004": "IAVB", "164909002": "LBBB",
    "59118001": "RBBB", "713427006": "RBBB", "284470004": "PAC", "63593006": "PAC",
    "427172004": "PVC", "17338001": "PVC", "164884008": "PVC",
    "429622005": "STD", "164931005": "STE",
}
CPSC_PRIORITY = ["AF", "STE", "STD", "PVC", "PAC", "RBBB", "LBBB", "IAVB", "NSR"]
DX_PAT = re.compile(r"#\s*Dx:\s*(.*)")


def primary_of(mapped):
    for c in CPSC_PRIORITY:
        if c in mapped:
            return c
    return None


def main():
    import glob as _g
    heas = sorted(Path(p) for p in _g.glob(RAW_GLOB))
    print(f"raw hea: {len(heas)}")
    rec = {}
    for h in heas:
        txt = h.read_text(encoding="ascii", errors="ignore")
        m = DX_PAT.search(txt)
        codes = [c.strip() for c in m.group(1).split(",")] if m else []
        codes = [c for c in codes if c]
        mapped = sorted({CPSC_MAP[c] for c in codes if c in CPSC_MAP})
        rec[h.stem] = dict(codes=codes, mapped=mapped, primary=primary_of(mapped))

    # canonical 顺序(data/cpsc 类目录序+文件名序, 与加载器/预测 y_true 同序)
    import os
    rows, ys, ids, mismatch = [], [], [], 0
    split_counts = {}
    for split in ("train", "val", "test"):
        root = ROOT / "data/cpsc" / split
        n_s = 0
        for cls in sorted(d for d in os.listdir(root) if (root / d).is_dir()):
            for fn in sorted(os.listdir(root / cls)):
                if not fn.endswith(".npy"):
                    continue
                rid = fn[:-4]
                r = rec.get(rid)
                if r is None or r["primary"] is None:
                    mismatch += 1
                    rows.append((rid, split, cls, None, "", "", [0] * 9))
                    ys.append([0] * 9)
                    ids.append(rid)
                    continue
                if r["primary"] != cls:
                    mismatch += 1
                mh = [1 if c in r["mapped"] else 0 for c in CPSC_CLASSES]
                rows.append((rid, split, cls, r["primary"], ";".join(r["codes"]),
                             ";".join(r["mapped"]), mh))
                ys.append(mh)
                ids.append(rid)
                n_s += 1
        split_counts[split] = n_s
    ys = np.asarray(ys, dtype=np.int8)
    print(f"aligned {len(rows)} (splits {split_counts}), primary!=dir: {mismatch}")
    assert mismatch == 0, "主类与目录放置不一致, 停止(须先定因)"

    OUTD.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow(["record_id", "split", "dir_class", "primary_class",
                     "all_snomed_codes", "mapped_classes", *CPSC_CLASSES])
    for rid, sp, dc, pr, codes, mp, mhs in rows:
        writer.writerow([rid, sp, dc, pr, codes, mp, *mhs])
    (OUTD / "cpsc_multilabel.csv").write_text(buf.getvalue(), encoding="utf-8", newline="")
    np.savez_compressed(OUTD / "cpsc_multilabel.npz", multihot=ys,
                        record_ids=np.array(ids), classes=np.array(CPSC_CLASSES))
    n_multi = int((ys.sum(1) >= 2).sum())
    percls_any = {c: int(ys[:, i].sum()) for i, c in enumerate(CPSC_CLASSES)}
    percls_pri = Counter(r[3] for r in rows)
    summary = dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), n=len(rows),
                   splits=split_counts, n_multilabel=n_multi,
                   pct_multilabel=round(100 * n_multi / len(rows), 2),
                   per_class_any=percls_any,
                   per_class_primary=dict(percls_pri),
                   note="顺序=canonical(类目录序+文件名序, 与预测 y_true 同序); "
                        "主类口径=CPSC_PRIORITY 与 prepare_cpsc_chapman 一致(校验 0 不一致)")
    (OUTD / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
