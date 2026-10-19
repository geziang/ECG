# -*- coding: utf-8 -*-
"""W9 正式投稿 FT10 补齐主链(主机A, 任务书 2026-10-19 冻结协议)。

阶段: preflight -> smoke -> run(双车道错峰) ; finalize 由 run_w9_finalize.py 承担。

矩阵: 任务书§三 27 次 + 用户 10-19 裁决扩充 m5-ptbxl×3 = 30 次
      (M5 PTB-XL FT10 全库为零, 收官门§七-7 要求 M5 三域闭合, 见 events.md)。

协议: 逐字沿用 run_ft.py 既有 FT10 —— fraction 0.1 / seed{0,2,4} / epoch100 /
      bs128 / Adam lr1e-4 / workers4 / val-loss 选点 / test 单评 / --trc 0 /
      不调参不换种子; 只复用冻结 checkpoint, 禁新 PT。

纪律: 幂等(账本JSON+metrics.json+四预测工件+元数据一致才跳过); 逐任务原子落账
      (每任务独立 JSON, 单写者线程全量重生成 results.csv, 无锁并发追加);
      失败即停相应车道并写 events.md; smoke 不碰 test。

用法(conda DL 解释器, 仓库根为 cwd):
    python runlog/W9/run_w9_submission_ft10.py --stages preflight
    python runlog/W9/run_w9_submission_ft10.py --stages smoke
    python runlog/W9/run_w9_submission_ft10.py --stages run              # 双车道, 错峰600s
    python runlog/W9/run_w9_submission_ft10.py --stages run --lanes 1    # IO 挤压时退单车道
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.pathguard import open_out

PY = sys.executable
OUTD = ROOT / "runlog/W9"
LOGD = OUTD / "logs"
LEDGER = OUTD / "ledger"
PROTO = "w9-submission-ft10-20261019"
SEEDS = [0, 2, 4]
NC = {"ptbxl": 5, "cpsc": 9, "chapman": 4}
SPLIT_FROZEN = {  # 任务书§4.2
    "ptbxl": "5ca8de4ecd672a1e",
    "cpsc": "7808e41777518f4c",
    "chapman": "09783cad0542e5d1",
}
CPSC_TEST_N = 1385  # W5A 修正分区
HPREF = "W9任务书§二/§四(w9-submission-ft10-20261019): run_ft.py 逐字沿用, lr1e-4/ep100/bs128/workers4/trc0"
PRETRAIN_SOURCE = "pt_pretrain_nfh"
RESULTS_COLS = ["ts", "method", "seed", "pretrain_source", "eval", "dataset",
                "label_fraction", "auroc", "auprc", "git_sha", "checkpoint_sha256",
                "split_sha", "protocol_id", "hparams_ref", "wall_s", "status"]

# ---- 方法 -> checkpoint 路径(跨批次目录) ----
def _ck_simclr(seed):
    return ROOT / f"checkpoint/confirm/simclr_seed{seed}"

def _ck_simph(seed):
    return ROOT / f"checkpoint/w8_simph/simph_seed{seed}"

def _ck_clocs(seed):
    return ROOT / f"checkpoint/confirm/clocs_seed{seed}"

def _ck_tsr(seed):
    return ROOT / f"checkpoint/w8_tsr/tsr_seed{seed}"

def _ck_m2(seed):
    return ROOT / ("checkpoint/w6_dv/seed0" if seed == 0
                   else f"checkpoint/w7_dv/seed{seed}")

def _ck_m5(seed):
    return ROOT / f"checkpoint/w8_noiseiso/seed{seed}"

METHODS = {
    "simclr": dict(part="P0", ck=_ck_simclr, datasets=["chapman"],
                   label="S1 SimCLR"),
    "simph": dict(part="P0", ck=_ck_simph, datasets=["chapman"],
                  label="SimCLR+NSTDB"),
    "clocs": dict(part="P0", ck=_ck_clocs, datasets=["cpsc", "chapman"],
                  label="S2 CLOCS"),
    "tsr": dict(part="P0", ck=_ck_tsr, datasets=["chapman"],
                label="TSR pretext"),
    "m2": dict(part="P1", ck=_ck_m2, datasets=["cpsc", "chapman"],
               label="M2 LFBT+Beat-RR"),
    "m5": dict(part="P1", ck=_ck_m5, datasets=["cpsc", "chapman"],
               label="M5 LFBT+NSTDB"),
}
# 用户 10-19 裁决: M5 PTB-XL FT10 全库为零, 补 3 次闭合三域(同冻结协议, SHA 已在§4.3 表)
MATRIX_EXTENSION = [("m5", "ptbxl", "P1x")]

SHA_FROZEN = {  # 任务书§4.3 (encoder_group.pth 完整 SHA256)
    ("simclr", 0): "cc28ed3a955490014cb582c9fb75193140b9ca5d083e3f783963b76c80d2f103",
    ("simclr", 2): "e4b431ba5c601bd52de3ad0e9b70d003d6840b5b8ce81409b50cae2b44d8b677",
    ("simclr", 4): "5958c81aeef97f9a35144740bec47216f5b3cd8e023737f19dbdca85c328fc30",
    ("simph", 0): "b4babeb6afd6abd0737e9ed4ece96572663f6ecf70bf6e4aa9fa1b71b65e8253",
    ("simph", 2): "9977a453a579d749bee0c6aaae06bfb51c0348e26aa39bbfe6b63a19d7f549f3",
    ("simph", 4): "c506166e327f5613f0ffdf3193c5dd69552570bbe4bc40d04ba6d40728c043b0",
    ("clocs", 0): "c1f139d98059812aeee6453c5c7ec9cb2e99d57f338178833cd6e0cd159574d5",
    ("clocs", 2): "69bc974a433ae50222178a4d084e07ed175254a3100034c7ad06451d2c4f11c1",
    ("clocs", 4): "914ed37ad35e05d70cfb00c217f5f622dfd6eef0441507ba02a5da968f13e995",
    ("tsr", 0): "bbb501f11e3da447d3da735eeedf51b96ae806e4d63298470aa63f3fbd700bfd",
    ("tsr", 2): "9c9560444831d2d1c12dee948812bbdc91e25a280b363aae492b61e4717ff27e",
    ("tsr", 4): "b5fe62d2c6e49b0ca6ba6901ac545e8dee354523f4af3a37c906442b2c5f6616",
    ("m2", 0): "123c4018768a9549dff7c418ebf61c1e1b2bf858a06836c146c754b5cb6ef726",
    ("m2", 2): "62c4209d1c8a319dfca9f6b6284a6a06850dc1953e6eff56c286075f3af793f1",
    ("m2", 4): "69272f12f516cd140e802dbf27eb57e1625f8be19e62ec853098a916d48acbe2",
    ("m5", 0): "6883e5254be34895f8bc12859e3cd9f58fa6948a6b9aa629fe470b4641d9c4e4",
    ("m5", 2): "6747cb1b114df946fcb846e138ff447dd173ef271d735c0575419032b1d6809e",
    ("m5", 4): "8685ce9dc70d530db228424ba4eff2282eac7940230a16a665272bf12cf562fd",
}

# config.json 语义门(§4.3: SHA 匹配且配置语义须与矩阵一致; 公共项另查)
CONFIG_SEM = {
    "simclr": {"loss_mode": "simclr", "nstdb_aug": 0.0},
    "simph": {"loss_mode": "simclr", "nstdb_aug": 0.5},
    "clocs": {"loss_mode": "clocs"},
    "tsr": {"loss_mode": "tsr"},
    "m2": {"loss_mode": "bt", "dualview_weight": 0.01, "rr_weight": 0.1},
    "m5": {"loss_mode": "bt", "nstdb_aug": 0.5, "nstdb_offset": "first_half",
           "dualview_weight": 0.0, "rr_weight": 0.0},
}
SEM_ABSENT_OK = {"nstdb_aug", "dualview_weight", "rr_weight"}  # 缺失按 0 处理的字段
SEM_STRING_FIELDS = {"nstdb_offset", "loss_mode"}

PRINT_LOCK = threading.Lock()
MERGE_LOCK = threading.Lock()
_STOP = {"reason": None}
RUN_SHA = {"v": None}  # 正式运行统一盖章 preflight 时的 git SHA(§七-6 与 preflight 一致)


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    with PRINT_LOCK:
        print(line, flush=True)
        LOGD.mkdir(parents=True, exist_ok=True)
        with open_out(LOGD, "chain.log", mode="a", encoding="utf-8") as f:
            f.write(line + "\n")


def event(msg):
    OUTD.mkdir(parents=True, exist_ok=True)
    with open_out(OUTD, "events.md", mode="a", encoding="utf-8") as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} {msg}\n")
    log("EVENT: " + msg)


def git_sha():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ================================================================ 任务矩阵
def build_cells():
    """静态矩阵(§三 27 + 用户裁决扩充 3)的单元格层, 顺序遵§九:
    CLOCS-CPSC 先行 -> 四法 Chapman -> M2/M5 CPSC+Chapman -> m5-ptbxl(扩充)。"""
    cells = []
    cells.append(("clocs", "cpsc", "P0"))
    for m in ("simclr", "simph", "clocs", "tsr"):
        cells.append((m, "chapman", "P0"))
    for m in ("m2", "m5"):
        for ds in ("cpsc", "chapman"):
            cells.append((m, ds, "P1"))
    for m, ds, part in MATRIX_EXTENSION:
        cells.append((m, ds, part))
    assert len(cells) == 10, len(cells)  # 10 单元格 × 3 seeds = 30
    return cells


def all_seed_tasks():
    out = []
    idx = 0
    for (m, ds, part) in build_cells():
        for seed in SEEDS:
            ck = METHODS[m]["ck"](seed)
            out.append(dict(idx=idx, part=part, method=m, dataset=ds, seed=seed,
                            ckpt=ck, tag=f"{m}_{ds}_s{seed}",
                            sha=SHA_FROZEN[(m, seed)],
                            split_sha=SPLIT_FROZEN[ds], nc=NC[ds]))
            idx += 1
    assert len(out) == 30, len(out)
    tags = [t["tag"] for t in out]
    assert len(set(tags)) == 30, "tag 重复"
    from collections import Counter
    pc = Counter(t["part"] for t in out)
    assert pc["P0"] == 15 and pc["P1"] == 12 and pc["P1x"] == 3, pc
    return out


# ================================================================ preflight
def split_lines(ds):
    """与 run_abc_chain.py::split_sha_of 同口径: 类目录序+文件名序。"""
    lines = []
    for split in ("train", "val", "test"):
        root = ROOT / "data" / ds / split
        lines.append(f"[{split}]")
        for cls in sorted(d for d in os.listdir(root) if (root / d).is_dir()):
            for fn in sorted(os.listdir(root / cls)):
                if fn.endswith(".npy"):
                    lines.append(f"{cls}/{fn}")
    return lines


def split_sha_of(ds):
    lines = split_lines(ds)
    sha = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]
    counts = {}
    cur = None
    for l in lines:
        if l.startswith("["):
            cur = l.strip("[]")
            counts[cur] = 0
        else:
            counts[cur] += 1
    return sha, counts, lines


def check_config_semantics(method, cfg):
    """§4.3: 架构/语料/方法标签与矩阵一致。返回 (ok, 差异列表)。"""
    diffs = []
    a = cfg.get("args", cfg)
    # 公共项: trc=0 / blur_pool=0 / pool_power=0 / 8 导联 / NFH 语料
    if float(a.get("trc", 0) or 0) != 0.0:
        diffs.append(f"trc={a.get('trc')}!=0")
    if float(a.get("blur_pool", 0) or 0) != 0.0:
        diffs.append(f"blur_pool={a.get('blur_pool')}!=0")
    try:
        if float(a.get("pool_power", 0) or 0) != 0.0:
            diffs.append(f"pool_power={a.get('pool_power')}!=0")
    except (TypeError, ValueError):
        diffs.append(f"pool_power 非数值: {a.get('pool_power')!r}")
    if int(a.get("num_leads", 8)) != 8:
        diffs.append(f"num_leads={a.get('num_leads')}!=8")
    dd = str(a.get("data_dir", "")).replace("\\", "/")
    if not dd.endswith("pt_pretrain_nfh"):
        diffs.append(f"data_dir={a.get('data_dir')} 非 NFH 语料")
    # 方法专属项
    for k, want in CONFIG_SEM[method].items():
        got = a.get(k, None)
        if got is None:
            if k in SEM_ABSENT_OK and float(want) == 0.0:
                continue
            diffs.append(f"{k} 缺失(期望 {want!r})")
            continue
        if k in SEM_STRING_FIELDS:
            if str(got) != str(want):
                diffs.append(f"{k}={got!r}!={want!r}")
        else:
            try:
                if abs(float(got) - float(want)) > 1e-9:
                    diffs.append(f"{k}={got}!={want}")
            except (TypeError, ValueError):
                diffs.append(f"{k} 非数值: {got!r}")
    return (not diffs), diffs


def stage_preflight():
    OUTD.mkdir(parents=True, exist_ok=True)
    rep = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "protocol_id": PROTO}
    fails = []

    # 4.1 代码与环境
    rep["git_sha"] = git_sha()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                            "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    rep["git_tracked_clean"] = not dirty
    if dirty:
        fails.append(f"tracked files 不 clean: {dirty!r}")

    import torch
    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                               "--format=csv,noheader"], capture_output=True,
                              text=True).stdout.strip()
    rep["gpu"] = gpu_name
    if "RTX 4090" not in gpu_name:
        fails.append(f"GPU 非 RTX 4090: {gpu_name}")
    rep["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        fails.append("CUDA 不可用")
    rep["python"] = sys.version.split()[0]
    rep["torch"] = torch.__version__
    rep["cuda"] = torch.version.cuda
    rep["cudnn"] = torch.backends.cudnn.version()
    if rep["python"] != "3.11.9":
        fails.append(f"Python {rep['python']} != 3.11.9(先停下记录, 不自行升降级)")
    if rep["torch"] != "2.5.1+cu121":
        fails.append(f"PyTorch {rep['torch']} != 2.5.1+cu121")

    # 4.2 数据划分
    manifest = {"ts": rep["ts"], "protocol_id": PROTO, "datasets": {}}
    for ds, want_sha in SPLIT_FROZEN.items():
        sha, counts, lines = split_sha_of(ds)
        ok = sha == want_sha
        rep[f"split_sha.{ds}"] = dict(got=sha, want=want_sha, ok=ok,
                                      counts=counts)
        if not ok:
            fails.append(f"{ds} split SHA {sha} != 冻结 {want_sha}")
        if ds == "cpsc" and counts.get("test") != CPSC_TEST_N:
            fails.append(f"cpsc test={counts.get('test')} != {CPSC_TEST_N}(须 W5A 修正分区)")
        cls_dirs = sorted(d for d in os.listdir(ROOT / "data" / ds / "train")
                          if (ROOT / "data" / ds / "train" / d).is_dir())
        if len(cls_dirs) != NC[ds]:
            fails.append(f"{ds} 类数 {len(cls_dirs)} != {NC[ds]}")
        so = OUTD / "sample_order"
        so.mkdir(parents=True, exist_ok=True)
        with open_out(so, f"{ds}_sample_order.txt", mode="w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        manifest["datasets"][ds] = dict(split_sha=sha, counts=counts,
                                        n_classes=len(cls_dirs),
                                        classes=cls_dirs,
                                        sample_order_file=f"sample_order/{ds}_sample_order.txt")
    _atomic_write_json(OUTD / "data_manifest.json", manifest)

    # 4.3 checkpoint SHA256 + config 语义
    ckm = {}
    for (m, seed), want in sorted(SHA_FROZEN.items()):
        ck = METHODS[m]["ck"](seed)
        enc = ck / "encoder_group.pth"
        if not enc.exists():
            fails.append(f"{m} s{seed} 缺文件: {enc}")
            ckm[f"{m}_s{seed}"] = dict(path=str(enc.relative_to(ROOT)), exists=False)
            continue
        got = sha256_of(enc)
        entry = dict(path=str(enc.relative_to(ROOT)), sha256=got, want=want,
                     sha_ok=(got == want))
        cfgp = ck / "config.json"
        if cfgp.exists():
            cfg = json.loads(cfgp.read_text(encoding="utf-8"))
            sem_ok, diffs = check_config_semantics(m, cfg)
            entry["config_semantics_ok"] = sem_ok
            entry["config_diffs"] = diffs
            if not sem_ok:
                fails.append(f"{m} s{seed} config 语义不匹配: {diffs}")
        else:
            fails.append(f"{m} s{seed} 缺 config.json")
        if not entry["sha_ok"]:
            fails.append(f"{m} s{seed} SHA {got} != 冻结 {want}")
        ckm[f"{m}_s{seed}"] = entry
    _atomic_write_json(OUTD / "checkpoints_manifest.json",
                       dict(ts=rep["ts"], entries=ckm))

    # 静态矩阵落盘
    rows = ["idx,part,method,dataset,seed,ckpt_relpath,checkpoint_sha256,split_sha,num_classes"]
    for t in all_seed_tasks():
        rows.append(",".join(str(x) for x in [
            t["idx"], t["part"], t["method"], t["dataset"], t["seed"],
            str(t["ckpt"].relative_to(ROOT)), t["sha"], t["split_sha"], t["nc"]]))
    (OUTD / "task_matrix.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rep["n_tasks"] = len(all_seed_tasks())
    rep["verdict"] = "PASS" if not fails else "FAIL"
    rep["failures"] = fails
    _atomic_write_json(OUTD / "preflight.json", rep)
    for k, v in sorted(rep.items()):
        if k.startswith("split_sha"):
            log(f"preflight {k}: {v}")
    log(f"preflight verdict={rep['verdict']} fails={len(fails)}")
    for f_ in fails:
        log(f"  FAIL: {f_}")
    if fails:
        event("preflight FAIL: " + "; ".join(fails))
        raise SystemExit(1)
    event(f"preflight PASS (git {rep['git_sha']}, GPU ok, split SHA 3/3, ckpt SHA 18/18, config 语义 18/18, 矩阵 {rep['n_tasks']} 次)")


# ================================================================ smoke(不碰 test)
def stage_smoke():
    """§五-6: 每方法×数据集组合用 seed0 ckpt 做 checkpoint 加载 + 1 train batch
    (前向+反向) + 1 val batch(前向) + 输出维度检查。禁止迭代 test_loader。"""
    import numpy as np
    import torch
    import torch.nn as nn
    from data_utils.cls_datasets import get_data_loaders
    from models.mbn import MultiBranchNet
    from data_utils.seed_utils import set_seed

    combos = []
    for m, info in METHODS.items():
        for ds in info["datasets"]:
            combos.append((m, ds))
    for m, ds, _ in MATRIX_EXTENSION:
        if (m, ds) not in combos:
            combos.append((m, ds))

    results = []
    fail = False
    for m, ds in combos:
        set_seed(0)
        ck = METHODS[m]["ck"](0)
        nc = NC[ds]
        model = None
        try:
            model = MultiBranchNet(nc, checkpoint=ck / "encoder_group.pth",
                                   blur_pool=0, pool_power=0.0, trc=0)
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(dev)
            tr, vl, _te = get_data_loaders(str(ROOT / f"data/{ds}"), 2, 0,
                                           train_ratio=0.1, seed=0)
            xb, yb = next(iter(tr))
            out = model(xb.to(dev))
            loss = nn.functional.cross_entropy(out, yb.to(dev))
            loss.backward()
            xv, yv = next(iter(vl))
            outv = model(xv.to(dev))
            assert tuple(out.shape) == (2, nc), f"train 输出维度 {tuple(out.shape)}!=({2},{nc})"
            assert tuple(outv.shape) == (2, nc), f"val 输出维度 {tuple(outv.shape)}!=({2},{nc})"
            assert torch.isfinite(loss).all(), "train loss 非有限"
            ok = True
            detail = f"loss={loss.item():.4f}"
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append(dict(method=m, dataset=ds, seed=0, ok=ok, detail=detail))
        log(f"smoke {m}-{ds}: {'OK' if ok else 'FAIL'} ({detail})")
        if not ok:
            fail = True
        if model is not None:
            del model
        torch.cuda.empty_cache()
    _atomic_write_json(OUTD / "smoke.json",
                       dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), results=results))
    if fail:
        event("smoke FAIL(见 smoke.json)")
        raise SystemExit(1)
    event(f"smoke PASS ({len(combos)}/{len(combos)} 组合, 未触碰 test)")


# ================================================================ run
class LaneStop(Exception):
    pass


def task_done(t):
    """幂等跳过: 账本JSON(ok) + metrics.json + 四预测工件 + 元数据一致。"""
    lj = LEDGER / f"{t['tag']}.json"
    if not lj.exists():
        return False
    try:
        row = json.loads(lj.read_text(encoding="utf-8"))
    except Exception:
        return False
    if row.get("status") != "ok":
        return False
    mdir = OUTD / "ft" / t["tag"]
    if not (mdir / "metrics.json").exists():
        return False
    pdir = OUTD / "predictions" / t["tag"]
    for f_ in ("y_true.npy", "y_pred.npy", "y_prob.npy", "metrics_ext.json"):
        if not (pdir / f_).exists():
            return False
    try:
        m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
        if abs(float(m["auroc"]) - float(row["auroc"])) > 1e-12:
            return False
        if abs(float(m["auprc"]) - float(row["auprc"])) > 1e-12:
            return False
        me = json.loads((pdir / "metrics_ext.json").read_text(encoding="utf-8"))
        meta = me.get("metadata", {})
        if meta.get("protocol_id") != PROTO or meta.get("seed") != t["seed"]:
            return False
        if meta.get("checkpoint_sha256") != t["sha"]:
            return False
        if int(meta.get("num_classes", -1)) != t["nc"]:
            return False
    except Exception:
        return False
    return True


def validate_task_outputs(t):
    """非零返回码之外的硬校验: 指标有限/范围, 预测维度/有限/行和, 元数据一致。"""
    import numpy as np
    mdir = OUTD / "ft" / t["tag"]
    pdir = OUTD / "predictions" / t["tag"]
    m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
    for k in ("auroc", "auprc"):
        v = float(m[k])
        if not (np.isfinite(v) and 0.0 <= v <= 1.0):
            raise ValueError(f"metrics.{k}={v} 非有限/越界")
    yt = np.load(pdir / "y_true.npy")
    yp = np.load(pdir / "y_pred.npy")
    pr = np.load(pdir / "y_prob.npy")
    _, counts, _ = split_sha_of(t["dataset"])
    n_test = counts["test"]
    if yt.shape[0] != n_test:
        raise ValueError(f"y_true n={yt.shape[0]} != test {n_test}")
    if pr.shape != (n_test, t["nc"]):
        raise ValueError(f"y_prob 形状 {pr.shape} != ({n_test},{t['nc']})")
    if not (np.isfinite(pr).all() and np.isfinite(yt).all()):
        raise ValueError("预测含非有限值")
    if not np.allclose(pr.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("y_prob 行和不≈1")
    if not (yp == pr.argmax(axis=1)).all():
        raise ValueError("y_pred != argmax(y_prob)")
    me = json.loads((pdir / "metrics_ext.json").read_text(encoding="utf-8"))
    meta = me.get("metadata", {})
    if meta.get("protocol_id") != PROTO:
        raise ValueError("metrics_ext protocol_id 不符")
    if meta.get("checkpoint_sha256") != t["sha"]:
        raise ValueError("metrics_ext checkpoint_sha256 不符")
    # 门5 独立复算: 从落盘预测重算指标须与 metrics.json 一致(>1e-6 即失败)。
    # 注: metrics_ext.macro_ap 系无ties处理的手写 AP, 与 sklearn 存在 ~1e-5 固有差,
    # 不得拿它与 sklearn auprc 做 1e-6 等价断言(20:41 假阳性根因), 只做 1e-3 粗检。
    from sklearn.metrics import average_precision_score, roc_auc_score
    y1h = np.eye(pr.shape[1])[yt]
    rc_auroc = roc_auc_score(y1h, pr, average="macro")
    rc_auprc = average_precision_score(y1h, pr, average="macro")
    if abs(rc_auroc - float(m["auroc"])) > 1e-6 or abs(rc_auprc - float(m["auprc"])) > 1e-6:
        raise ValueError("metrics.json 与预测独立复算不一致(>1e-6)")
    if abs(float(me.get("macro_ap", float("nan"))) - rc_auprc) > 1e-3:
        raise ValueError("metrics_ext.macro_ap 偏离复算 auprc >1e-3")
    return m


def merge_results():
    """单写者全量重生成 results.csv(原子替换), 行序=任务矩阵序。"""
    with MERGE_LOCK:
        rows = []
        for t in all_seed_tasks():
            lj = LEDGER / f"{t['tag']}.json"
            if lj.exists():
                row = json.loads(lj.read_text(encoding="utf-8"))
                rows.append(",".join(_csv_cell(row.get(c, "")) for c in RESULTS_COLS))
        tmp = OUTD / "results.csv.tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(RESULTS_COLS) + "\n")
            f.write("\n".join(rows) + ("\n" if rows else ""))
        os.replace(tmp, OUTD / "results.csv")


def _csv_cell(v):
    s = str(v)
    if any(c in s for c in ",\"\n"):
        return '"' + s.replace('"', '""') + '"'
    return s


def try_revalidate(t, lane):
    """§五-1 跳过规则的复验路径: metrics.json+四工件齐全且通过(修复后的)校验
    → 不重训直接入账。覆盖账本被历史假阳性标 invalid 但产物真实的任务。"""
    mdir = OUTD / "ft" / t["tag"]
    pdir = OUTD / "predictions" / t["tag"]
    need = [mdir / "metrics.json", pdir / "y_true.npy", pdir / "y_pred.npy",
            pdir / "y_prob.npy", pdir / "metrics_ext.json"]
    if not all(p.exists() for p in need):
        return False
    try:
        m = validate_task_outputs(t)
    except Exception:
        return False
    logf = LOGD / f"{t['tag']}.log"
    try:
        wall = round(os.path.getmtime(pdir / "metrics_ext.json")
                     - os.path.getctime(logf), 1)
    except OSError:
        wall = -1.0
    row = dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), method=t["method"],
               seed=t["seed"], pretrain_source=PRETRAIN_SOURCE, eval="ft10",
               dataset=t["dataset"], label_fraction="0.1",
               git_sha=RUN_SHA["v"] or git_sha(), checkpoint_sha256=t["sha"],
               split_sha=t["split_sha"], protocol_id=PROTO, hparams_ref=HPREF,
               auroc=f"{float(m['auroc']):.10f}", auprc=f"{float(m['auprc']):.10f}",
               wall_s=max(wall, 0.0), status="ok", revalidated=True,
               wall_s_note="复验入账, wall 按文件时间近似" if wall >= 0 else "复验入账")
    _atomic_write_json(LEDGER / f"{t['tag']}.json", row)
    merge_results()
    log(f"[L{lane}] REVALIDATED {t['tag']} auroc={row['auroc']} auprc={row['auprc']}"
        f" (产物复算一致, 免重训)")
    return True


def run_task(t, lane):
    if task_done(t):
        log(f"[L{lane}] skip(已完成): {t['tag']}")
        return
    if try_revalidate(t, lane):
        return
    mdir = OUTD / "ft" / t["tag"]
    pdir = OUTD / "predictions" / t["tag"]
    cmd = [PY, "-u", "run_ft.py",
           "--data-dir", f"data/{t['dataset']}",
           "--num-classes", str(t["nc"]),
           "--fraction", "0.1",
           "--checkpoint", str(t["ckpt"] / "encoder_group.pth"),
           "--model-dir", str(mdir),
           "--workers", "4", "--epochs", "100",
           "--batch-size", "128", "--learning-rate", "0.0001",
           "--seed", str(t["seed"]), "--trc", "0",
           "--extended-metrics", "1",
           "--save-predictions", f"runlog/W9/predictions/{t['tag']}",
           "--protocol-id", PROTO,
           "--checkpoint-sha", t["sha"],
           "--data-manifest-sha", t["split_sha"]]
    t0 = time.time()
    log(f"[L{lane}] START {t['tag']} ({t['ckpt'].relative_to(ROOT)})")
    with open_out(LOGD, f"{t['tag']}.log", mode="w", encoding="utf-8") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            cwd=str(ROOT)).returncode
    wall = round(time.time() - t0, 1)
    base = dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), method=t["method"],
                seed=t["seed"], pretrain_source=PRETRAIN_SOURCE, eval="ft10",
                dataset=t["dataset"], label_fraction="0.1",
                git_sha=RUN_SHA["v"] or git_sha(), checkpoint_sha256=t["sha"],
                split_sha=t["split_sha"], protocol_id=PROTO, hparams_ref=HPREF)
    if rc != 0:
        _atomic_write_json(LEDGER / f"{t['tag']}.json",
                           dict(base, wall_s=wall, status=f"rc{rc}"))
        merge_results()
        event(f"{t['tag']} 失败 rc={rc}, 车道{lane}停止(日志 logs/{t['tag']}.log)")
        raise LaneStop(f"{t['tag']} rc={rc}")
    try:
        m = validate_task_outputs(t)
    except Exception as e:
        _atomic_write_json(LEDGER / f"{t['tag']}.json",
                           dict(base, wall_s=wall, status=f"invalid({type(e).__name__})"))
        merge_results()
        event(f"{t['tag']} 产物校验失败({type(e).__name__}: {e}), 车道{lane}停止")
        raise LaneStop(f"{t['tag']} validation: {e}")
    row = dict(base, auroc=f"{float(m['auroc']):.10f}",
               auprc=f"{float(m['auprc']):.10f}",
               wall_s=wall, status="ok")
    _atomic_write_json(LEDGER / f"{t['tag']}.json", row)
    merge_results()
    log(f"[L{lane}] DONE {t['tag']} auroc={row['auroc']} auprc={row['auprc']} wall={wall}s")


def lane_worker(lane, tasks, stagger):
    if lane > 0 and stagger > 0:
        log(f"[L{lane}] 错峰等待 {stagger}s 后开工")
        time.sleep(stagger)
    for t in tasks:
        if _STOP["reason"]:
            log(f"[L{lane}] 全局停止({_STOP['reason']}), 退出")
            return
        try:
            run_task(t, lane)
        except LaneStop as e:
            _STOP["reason"] = str(e)
            log(f"[L{lane}] 失败即停: {e}(另一车道将随之退出)")
            return


def _pid_alive_python(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                         capture_output=True, text=True).stdout
    return "python" in out.lower() and str(pid) in out


def stage_run(lanes, stagger):
    # 进程自守: 已有活链条则拒绝二次启动(防双链条同任务并发互踩 model-dir)
    pidf = OUTD / "logs" / "run.pid"
    if pidf.exists():
        try:
            old = int(pidf.read_text().strip())
        except ValueError:
            old = -1
        if old > 0 and _pid_alive_python(old):
            log(f"已有 W9 链条在运行(pid={old}), 本次启动退出(幂等)")
            return
    LOGD.mkdir(parents=True, exist_ok=True)
    pidf.write_text(str(os.getpid()), encoding="ascii")
    try:
        _stage_run_inner(lanes, stagger)
    finally:
        try:
            pidf.unlink()
        except OSError:
            pass


def _stage_run_inner(lanes, stagger):
    pf = OUTD / "preflight.json"
    if pf.exists():
        RUN_SHA["v"] = json.loads(pf.read_text(encoding="utf-8")).get("git_sha")
        log(f"run: 行内 git_sha 统一盖章 preflight SHA {RUN_SHA['v']}")
    tasks = all_seed_tasks()
    n_done = sum(1 for t in tasks if task_done(t))
    log(f"run: 共 {len(tasks)} 任务, 已完成 {n_done}, 车道 {lanes}, 错峰 {stagger}s")
    event(f"run 启动: {len(tasks)} 任务/{lanes} 车道/错峰{stagger}s "
          f"(任务书27 + 用户裁决扩充m5-ptbxl×3)")
    if lanes == 1:
        lane_worker(0, tasks, 0)
    else:
        l0 = [t for i, t in enumerate(tasks) if i % 2 == 0]
        l1 = [t for i, t in enumerate(tasks) if i % 2 == 1]
        th0 = threading.Thread(target=lane_worker, args=(0, l0, 0))
        th1 = threading.Thread(target=lane_worker, args=(1, l1, stagger))
        th0.start()
        th1.start()
        th0.join()
        th1.join()
    n_ok = sum(1 for t in tasks
               if (LEDGER / f"{t['tag']}.json").exists()
               and json.loads((LEDGER / f"{t['tag']}.json").read_text(encoding="utf-8")).get("status") == "ok")
    log(f"run 结束: {n_ok}/{len(tasks)} ok")
    event(f"run 结束: {n_ok}/{len(tasks)} ok"
          + ("" if n_ok == len(tasks) else f", 停止原因: {_STOP['reason']}"))
    if n_ok != len(tasks):
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="run",
                    help="逗号分隔: preflight,smoke,run")
    ap.add_argument("--lanes", type=int, default=2, choices=(1, 2))
    ap.add_argument("--stagger", type=int, default=600,
                    help="车道错峰秒数(任务书§五-5 >=600)")
    a = ap.parse_args()
    for s in a.stages.split(","):
        s = s.strip()
        if s == "preflight":
            stage_preflight()
        elif s == "smoke":
            stage_smoke()
        elif s == "run":
            stage_run(a.lanes, a.stagger)
        else:
            raise SystemExit(f"未知阶段: {s}")


if __name__ == "__main__":
    main()
