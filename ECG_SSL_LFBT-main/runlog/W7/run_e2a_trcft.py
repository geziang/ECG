# -*- coding: utf-8 -*-
"""W7 E2a: FT期TRC(参数高效适配)——冻结 conv 骨干, 只训 零初始化TRC + 线性头。

对象: {c1, c2} × {ptbxl, cpsc} × FT10 × seeds{0,2,4} = 24 跑(纯下游, 无预训练)。
- c1(trc=0): 每 lead 的 VGG model=[b1..b5, pool] 在 pool 前插零初始化 GRN1D → 可训练参数仅 TRC+头;
- c2(trc=1): model 已含预训练 GRN, 全部冻结, 只解冻 GRN 参数+头(=TRC 作为适配器)。
协议: 与 full-FT10 逐位同款(fraction0.1, epochs100, bs128, lr1e-4, CE), 仅冻结面不同。
对照(引用不重跑): LP = W5(cpsc)/W2(ptbxl) 账本; full-FT10 = W6 ft10grid/ft2040 账本。
门(队列 Q6c 预注册): vs LP 3-seed 同向 + mean≥+0.5pt; 同时报告与 full-FT10 的收敛比。
账本: runlog/W7/e2a_trcft.csv + e2a_summary.md; protocol_id=w7-e2a。
用法: python runlog/W7/run_e2a_trcft.py [--dry-run] [--limit-secs N]
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PY = "C:/Users/admin/.conda/envs/DL/python.exe"
OUTD = ROOT / "runlog/W7"
LOGD = OUTD / "logs"
OUT = OUTD / "e2a_trcft.csv"
JOB = {"ptbxl": 5, "cpsc": 9}
TRC_OF = {"c1": 0, "c2": 1}
SEEDS = [0, 2, 4]


def ck_of(kind, seed):
    return ROOT / f"checkpoint/confirm/{kind}_seed{seed}/encoder_group.pth"


DRIVER = r'''
# -*- coding: utf-8 -*-
import argparse, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import sys
ROOT = Path(r"%ROOT%")
sys.path.insert(0, str(ROOT))
from data_utils.cls_datasets import get_data_loaders
from data_utils.seed_utils import set_seed
from models.vgg_1d import VGG16, GRN1D
from utils.checkpoint import load_torch_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir"), ap.add_argument("--num-classes", type=int)
ap.add_argument("--checkpoint"), ap.add_argument("--trc", type=int)
ap.add_argument("--seed", type=int), ap.add_argument("--model-dir")
ap.add_argument("--fraction", type=float, default=0.1)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch-size", type=int, default=128)
ap.add_argument("--learning-rate", type=float, default=1e-4)
ap.add_argument("--workers", type=int, default=0)
ap.add_argument("--head-init", default="")
A = ap.parse_args()
set_seed(A.seed)
dev = "cuda"
mdir = Path(A.model_dir); mdir.mkdir(parents=True, exist_ok=True)

ck = load_torch_checkpoint(Path(A.checkpoint), map_location="cpu")
encs = []
for i in range(8):
    v = VGG16(ch_in=1, alpha=0.125, trc=1)
    sd = ck["backbone_state_dict_list"][i] if "backbone_state_dict_list" in ck else ck["backbone_state_dict"][i].state_dict()
    missing, unexpected = v.load_state_dict(sd, strict=False)
    # c1(trc=0) ckpt 无 GRN 键 -> missing 含 model.5.gamma/beta = 零初始化 FT期TRC 注入(预期);
    # c2(trc=1) ckpt 有 GRN 键 -> 预训练 GRN 值加载, 作为可训练适配器。
    assert unexpected == [] and set(missing) <= {"fc.weight", "fc.bias", "model.5.gamma", "model.5.beta"}, missing
    encs.append(v.model.to(dev))  # [b1..b5, GRN1D, pool] (trc=1 结构; c1 ckpt 的 GRN 缺省=零初始化, 恰为 FT期TRC 注入)
for v in encs:
    for p in v.parameters():
        p.requires_grad = False
trc_params = []
for v in encs:
    g = v[5]  # GRN1D
    for p in g.parameters():
        p.requires_grad = True
        trc_params.append(p)

class Net(nn.Module):
    def __init__(self, encs):
        super().__init__()
        self.encs = encs
        self.fc = nn.Linear(512, A.num_classes).to(dev)
        if A.head_init:
            hsd = torch.load(A.head_init, map_location="cpu", weights_only=True)
            self.fc.weight.data = hsd["linear.weight"].to(dev)
            self.fc.bias.data = hsd["linear.bias"].to(dev)
    def forward(self, x):
        fs = [self.encs[j](x[:, [j], :]).squeeze(-1) for j in range(8)]
        return self.fc(torch.cat(fs, dim=1))

net = Net(encs).to(dev)
loss_fn = nn.CrossEntropyLoss().to(dev)
opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=A.learning_rate)
n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"trainable params: {n_train} (TRC {sum(p.numel() for p in trc_params)} + head)")

tr, va, te = get_data_loaders(A.data_dir, A.batch_size, A.workers, train_ratio=A.fraction, seed=A.seed)
best_val, best_state = float("inf"), None
for ep in range(A.epochs):
    net.train()
    for v in encs:  # 冻结骨干必须 eval: BN running stats 不得在 10% 子集上更新(12:56 发现的 bug, 已修)
        v.eval()
    for x, y in tr:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(); out = net(x); loss = loss_fn(out, y); loss.backward(); opt.step()
    net.eval(); vl = 0.0
    with torch.no_grad():
        for x, y in va:
            x, y = x.to(dev), y.to(dev)
            vl += loss_fn(net(x), y).item()
    vl /= max(len(va), 1)
    if vl < best_val:
        best_val = vl
        best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
    if (ep + 1) % 20 == 0:
        print(f"ep{ep+1} val_loss {vl:.4f}", flush=True)
net.load_state_dict(best_state)
net.eval()
ys, ps, pr = [], [], []
with torch.no_grad():
    for x, y in te:
        p = torch.softmax(net(x.to(dev)), dim=1).cpu().numpy()
        ys.append(y.numpy()); ps.append(p.argmax(1)); pr.append(p)
y = np.concatenate(ys); yp = np.concatenate(ps); prob = np.concatenate(pr)
from sklearn.metrics import roc_auc_score, average_precision_score
y1 = np.eye(A.num_classes)[y]
auroc = float(roc_auc_score(y1, prob, average="macro")); auprc = float(average_precision_score(y1, prob))
(mdir / "metrics.json").write_text(json.dumps(dict(auroc=auroc, auprc=auprc, trainable_params=n_train, best_val=best_val)), encoding="utf-8")
np.save(mdir / "y_true.npy", y); np.save(mdir / "y_pred.npy", yp); np.save(mdir / "y_prob.npy", prob)
print(f"TEST auroc={auroc:.4f} auprc={auprc:.4f}")
'''


def log(msg):
    print(f"[e2a {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def row_done(kind, ds, seed):
    if not OUT.exists():
        return False
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["ckpt"], r["downstream"], r["seed"]) == (kind, ds, str(seed)):
                return True
    return False


def record(kind, ds, seed, auroc, auprc, ntrain):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "fraction", "auroc", "auprc",
                        "trainable_params", "protocol_id"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), kind, seed, "trcft10", ds, "0.1",
                    f"{auroc:.4f}", f"{auprc:.4f}", ntrain, "w7-e2a"])
    log(f"  + e2a {kind}/{ds} s{seed}: {auroc:.4f}/{auprc:.4f}")


def run_one(kind, ds, seed, dry=False):
    if row_done(kind, ds, seed):
        log(f"e2a {kind}/{ds} s{seed}: 已有账, 跳过")
        return True
    import hashlib
    ck = ck_of(kind, seed)
    if not ck.exists():
        log(f"e2a {kind}/{ds} s{seed}: ckpt 缺失 {ck}")
        return False
    tag = f"e2a_{kind}_{ds}_trcft_seed{seed}"
    mdir = OUTD / "ft" / tag
    if dry:
        log(f"{tag}: DRY ok")
        return True
    from utils.pathguard import open_out
    hp = (ROOT / f"results/confirm/{kind}_{ds}_seed{seed}/classifier_best_ckpt.pth" if ds == "ptbxl"
          else ROOT / f"runlog/W5/feat/{kind}_{ds}_seed{seed}/classifier_best_ckpt.pth")
    if not hp.exists():
        log(f"{tag}: LP 头缺失 {hp}")
        return False
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "-c", DRIVER.replace("%ROOT%", str(ROOT)),
             "--data-dir", f"data/{ds}", "--num-classes", str(JOB[ds]),
             "--checkpoint", str(ck), "--trc", str(TRC_OF[kind]),
             "--seed", str(seed), "--model-dir", str(mdir),
             "--head-init", str(hp)],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
        # 逐记录预测落盘到 W7 预测目录(复用 runlog 产物规范)
        import shutil
        pdir = OUTD / "predictions" / tag
        pdir.mkdir(parents=True, exist_ok=True)
        for f_ in ("y_true.npy", "y_pred.npy", "y_prob.npy"):
            shutil.copyfile(mdir / f_, pdir / f_)
        record(kind, ds, seed, m["auroc"], m["auprc"], m["trainable_params"])
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def summary():
    import numpy as np
    ours = {}
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ours[(r["ckpt"], r["downstream"], r["seed"])] = float(r["auroc"])
    # LP 参照: cpsc=W5 lp_results; ptbxl=W2 stats per_seed 首位 AUROC 未知→用 seedext 账本
    lp_ref = {}
    p5 = ROOT / "runlog/W5/lp_results.csv"
    if p5.exists():
        with open(p5, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("downstream") == "cpsc" and r.get("ckpt") in ("c1", "c2"):
                    lp_ref[(r["ckpt"], "cpsc", r["seed"])] = float(r["auroc"])
    p6 = ROOT / "runlog/W6/seedext_results_L1.csv"
    if p6.exists():
        with open(p6, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("downstream") == "ptbxl" and r.get("eval") == "lp" and r.get("ckpt") in ("c1", "c2"):
                    lp_ref[(r["ckpt"], "ptbxl", r["seed"])] = float(r["auroc"])
    ft_ref = {}
    for pth, evf in ((ROOT / "runlog/W6/ft10grid_results_L1.csv", None),
                     (ROOT / "runlog/W6/ft2040_results.csv", "ft10")):
        if pth.exists():
            with open(pth, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("downstream") in JOB and r.get("ckpt") in ("c1", "c2"):
                        if r.get("eval") in ("ft10", None) or evf is None:
                            ft_ref[(r["ckpt"], r["downstream"], r["seed"])] = float(r["auroc"])
    lines = ["# E2a FT期TRC(参数高效适配)摘要", "",
             "| kind | domain | seed | LP | TRC-FT10 | Δ(vs LP) | full-FT10 | 收敛比 |", "|---|---|---|---|---|---|---|---|"]
    for kind in ("c1", "c2"):
        for ds in JOB:
            for s in map(str, SEEDS):
                k = (kind, ds, s)
                if k in ours and k in lp_ref:
                    d = (ours[k] - lp_ref[k]) * 100
                    ft = ft_ref.get(k)
                    conv = f"{(ours[k] - lp_ref[k]) / max(ft - lp_ref[k], 1e-9) * 100:.0f}%" if ft and ft != lp_ref[k] else "-"
                    lines.append(f"| {kind} | {ds} | {s} | {lp_ref[k]:.4f} | {ours[k]:.4f} | {d:+.2f}pt | "
                                 f"{ft:.4f} | {conv} |" if ft else
                                 f"| {kind} | {ds} | {s} | {lp_ref[k]:.4f} | {ours[k]:.4f} | {d:+.2f}pt | - | - |")
    got = [k for k in ours if k in lp_ref]
    if got:
        lines += ["", "## 门判定(vs LP, 3-seed 同向 + mean≥+0.5pt)", ""]
        for kind in ("c1", "c2"):
            for ds in JOB:
                ks = [k for k in got if k[0] == kind and k[1] == ds]
                if len(ks) < 3:
                    continue
                ds_ = [(ours[k] - lp_ref[k]) * 100 for k in ks]
                sign = all(d > 0 for d in ds_)
                ds_str = ", ".join("%+.2f" % d for d in ds_)
                lines.append(f"- {kind}/{ds}: Δ=[{ds_str}] mean={np.mean(ds_):+.2f}pt "
                             f"{'✅过门' if sign and np.mean(ds_) >= 0.5 else '未过门(如实记录)'}")
    (OUTD / "e2a_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("summary written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        for kind in ("c1", "c2"):
            for ds in JOB:
                for seed in SEEDS:
                    run_one(kind, ds, seed, dry=a.dry_run)
    summary()
