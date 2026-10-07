# -*- coding: utf-8 -*-
"""W7 E5: NSTDB 噪声注入预训练(独立轨道)——NFH 语料 + p=0.5 训练期真实噪声增强, seed0 单链。

协议: C1 逐位同款(NFH 100ep bs128 seed0 trc0 bt) + 唯一差异 --nstdb-aug 0.5(两视图独立注入)。
下游: 三域 LP; clean 门=对 C1 seed0 参照(cpsc=W5 修正分区, ptbxl/chapman=W2 confirm)掉幅≤0.30pt;
噪声评估: {bw,ma,em}×SNR{0,5,10}×三域(make_perturbed_real 同 W6 确定性口径) 对照 W6 robustness_real_c1.csv
seed0 同格; 门=clean 无伤 且 噪声格平均 ΔAUROC ≥+0.5pt(独立轨道, 判负即关不影响主线)。
账本: runlog/W7/e5_noiseaug_results.csv + e5_summary.md; protocol_id=w7-e5。
用法: python runlog/W7/run_e5_noiseaug_chain.py [--phase pt|all|eval] [--seed N]
  (seed 默认 0; seed{2,4}=Q8d 条件补做——seed0 过门才执行; 参照按同种子: LP=W2/W5 同种子行, 噪声格=W6 robustness_real_c1 同种子格;
   seed0 判定写 e5_summary.md, 其他种子写 e5_summary_seed{N}.md)
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.pathguard import open_out  # noqa: E402

PY = "C:/Users/admin/.conda/envs/DL/python.exe"
OUTD = ROOT / "runlog/W7"
LOGD = OUTD / "logs"
OUT = OUTD / "e5_noiseaug_results.csv"
CK = ROOT / "checkpoint/w7_e5/noiseaug_seed0"
DOMAINS = [("ptbxl", 5), ("cpsc", 9), ("chapman", 4)]
C1_REF = {("ptbxl"): 0.8907, ("cpsc"): 0.9455, ("chapman"): None}  # ptbxl/chapman=W2 s0; chapman 现查


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def git_sha():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def c1_refs():
    """C1 同种子三域 LP AUROC 参照 + W6 NSTDB 噪声格参照(同种子)。"""
    lp = {}
    with open(ROOT / "runlog/W2/confirm_results.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["ckpt"] == "c1" and r["seed"] == str(SEED) and r["eval"] == "lp":
                if r["downstream"] != "cpsc":  # cpsc 用 W5 修正分区
                    lp[r["downstream"]] = float(r["auroc"])
    with open(ROOT / "runlog/W5/lp_results.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["ckpt"] == "c1" and r["seed"] == str(SEED) and r["downstream"] == "cpsc":
                lp["cpsc"] = float(r["auroc"])
    noise = {}
    with open(ROOT / "runlog/W6/robustness_real_c1.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["seed"] == str(SEED):
                noise[(r["downstream"], r["noise"], r["intensity"])] = (float(r["auroc"]), float(r["auprc"]))
    return lp, noise


def row_done(ev, ds, extra=""):
    if not OUT.exists():
        return False
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["seed"], r["eval"], r["downstream"], r.get("cond", "")) == (str(SEED), ev, ds, extra):
                return True
    return False


def record(ev, ds, cond, auroc, auprc, cksha):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "cond", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "protocol_id"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), "c1na", str(SEED), ev, ds, cond,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha, "w7-e5"])
    log(f"  + e5 {ev}/{ds}/{cond}: {auroc:.4f}/{auprc:.4f}")


def run_pt():
    if (CK / "encoder_group.pth").exists() and (CK / "config.json").exists():
        log("e5 PT: 已完成, 跳过")
        return True
    CK.mkdir(parents=True, exist_ok=True)
    with open_out(LOGD, f"e5_pt_seed{SEED}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_pt.py", "--data-dir", "data/pt_pretrain_nfh",
             "--epochs", "100", "--batch-size", "128", "--workers", "4",
             "--seed", str(SEED), "--trc", "0", "--nstdb-aug", "0.5",
             "--resume", "--checkpoint-dir", str(CK)],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    ok = rc == 0 and (CK / "encoder_group.pth").exists()
    log(f"e5 PT: exit={rc} {'OK' if ok else 'FAIL'}")
    return ok


def run_lp(ds, nc):
    if row_done("lp", ds):
        log(f"e5 lp/{ds}: 已有账, 跳过")
        return True
    ckpt = CK / "encoder_group.pth"
    tag = f"e5_c1na_{ds}_lp_seed{SEED}"
    feat = OUTD / "feat" / tag
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--checkpoint", str(ckpt), "--feat-dir", str(feat),
             "--seed", str(SEED), "--workers", "6", "--trc", "0",
             "--extended-metrics", "1",
             "--save-predictions", str(OUTD / "predictions" / tag),
             "--protocol-id", "w7-e5"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
        record("lp", ds, "clean", m["auroc"], m["auprc"], sha256_of(ckpt))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def noise_eval():
    """冻结 c1na + 其 LP 头, W6 同款确定性噪声格评估, 对照 C1 的 W6 同格。"""
    if row_done("lpnoise", "done", "done"):
        return True
    import numpy as np
    import torch
    import torch.nn.functional as F
    from run_robustness import load_test, metrics_from_probs
    from run_robustness_nstdb import make_perturbed_real
    from models.vgg_1d import VGG16
    from models.linear import LinearClassifier
    from utils.checkpoint import load_torch_checkpoint
    dev = "cuda"
    ck = load_torch_checkpoint(CK / "encoder_group.pth", map_location="cpu")
    encs = []
    for i in range(8):
        v = VGG16(ch_in=1, alpha=0.125, trc=0)
        sd = ck["backbone_state_dict_list"][i] if "backbone_state_dict_list" in ck else ck["backbone_state_dict"][i].state_dict()
        v.load_state_dict(sd, strict=False)
        encs.append(torch.nn.Sequential(*list(v.children())[:-1]).to(dev).eval())

    def feats(x_t, bs=256):
        out = []
        with torch.no_grad():
            for b0 in range(0, x_t.shape[0], bs):
                xb = x_t[b0:b0 + bs]
                out.append(torch.cat([encs[j](xb[:, [j], :]).squeeze(-1) for j in range(8)], dim=1).cpu())
        return torch.cat(out)

    lp_ref, noise_ref = c1_refs()
    rows = []
    for ds, nc in DOMAINS:
        head = LinearClassifier(feat_dim=512, num_classes=nc).to(dev)
        head.load_state_dict(torch.load(OUTD / f"feat/e5_c1na_{ds}_lp_seed{SEED}/classifier_best_ckpt.pth",
                                        map_location=dev, weights_only=True))
        head.eval()
        x, y, _, _ = load_test(ds)
        for name in ("bw", "ma", "em"):
            for inten in ("0", "5", "10"):
                xp = make_perturbed_real(x, ds, "nstb_" + name, inten)
                with torch.no_grad():
                    probs = F.softmax(head(feats(torch.from_numpy(np.ascontiguousarray(xp)).cuda())), dim=1).cpu().numpy()
                au, ap_ = metrics_from_probs(y, probs, nc)
                ref = noise_ref.get((ds, "nstb_" + name, inten))
                rows.append((ds, f"nstb_{name}@{inten}", au, ap_, ref))
                record("lpnoise", ds, f"{name}@{inten}", au, ap_, sha256_of(CK / "encoder_group.pth"))
    # 判定
    clean_rows = []
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["eval"] == "lp":
                    clean_rows.append((r["downstream"], float(r["auroc"])))
    lines = ["# E5 噪声注入预训练判定(独立轨道, 队列 Q7 预注册)", ""]
    clean_ok = {}
    for ds, au in clean_rows:
        d = (au - lp_ref[ds]) * 100
        clean_ok[ds] = d
        lines.append(f"- clean {ds}: {lp_ref[ds]:.4f}→{au:.4f} ({d:+.2f}pt, 无伤容差 −0.30)")
    deltas = [(au - ref[0]) * 100 for (_, _, au, _, ref) in rows if ref]
    if deltas:
        import numpy as np
        m = float(np.mean(deltas))
        clean_all = all(v >= -0.30 for v in clean_ok.values())
        verdict = ("✅过门(clean 无伤 且 噪声均值≥+0.5pt)" if clean_all and m >= 0.5
                   else "❌判负关线(独立轨道, 不影响主线)" + ("(clean 受损)" if not clean_all else "(噪声增益不足)"))
        lines += ["", f"- 噪声格({len(deltas)}格) ΔAUROC 均值 vs C1(W6 同格) = {m:+.2f}pt",
                  f"- 判定: {verdict}"]
    sname = "e5_summary.md" if SEED == 0 else f"e5_summary_seed{SEED}.md"
    (OUTD / sname).write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"e5 summary written: {sname}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["pt", "all", "eval"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    SEED = args.seed
    CK = ROOT / f"checkpoint/w7_e5/noiseaug_seed{SEED}"
    log(f"E5 噪声注入预训练链启动(NFH+nstdb_aug0.5, seed={SEED})")
    if args.phase in ("pt", "all") and run_pt():
        if args.phase == "all":
            for ds, nc in DOMAINS:
                run_lp(ds, nc)
            noise_eval()
    elif args.phase == "eval":
        noise_eval()
    log("E5_CHAIN_END")
