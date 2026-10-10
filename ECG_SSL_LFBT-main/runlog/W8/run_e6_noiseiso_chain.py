# -*- coding: utf-8 -*-
"""W8 E6 w8-noiseiso: 噪声时间片段隔离复验链。

用法(conda DL 解释器, 仓库根为 cwd):
    python runlog/W8/run_e6_noiseiso_chain.py            # 全链 seeds{0,2,4}
    python runlog/W8/run_e6_noiseiso_chain.py --phase eval --seed 0

设计(任务书 2026-10-10 增补): 训练注入偏移限制每条 NSTDB 噪声记录前 50%
(--nstdb-aug 0.5 --nstdb-offset first_half), 评测偏移限制后 50%。若模型只是
"记住训练见过的噪声片段", 后半段(未见)上增益应消失; 门=隔离版增益方向与
W7 E5 原版一致(守住"非片段记忆")。

参照口径(关键): C1 冻结 checkpoint 用同一后半段偏移、同一噪声实现(同 rng 种子)
重评 27 格(ckpt=c1-lasteval 行), Δ=noiseiso−c1 同种子同格配对——不复用 W6 全域
偏移的旧 C1 数(那会混入评测侧口径变化)。

产物: runlog/W8/w8_noiseiso_results.csv(cond 列同 W7 E5 schema) + e6_summary.md。
幂等: PT 有 encoder+config 跳过; csv 已有行跳过。
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
from utils.pathguard import open_out

PY = sys.executable
OUTD = ROOT / "runlog/W8"
OUT = OUTD / "w8_noiseiso_results.csv"
LOGD = OUTD / "logs"
SEEDS = [0, 2, 4]
CKD = ROOT / "checkpoint/w8_noiseiso"
DOMAINS = [("ptbxl", 5), ("cpsc", 9), ("chapman", 4)]
HP = ("W8 E6: C1 协议 + --nstdb-aug 0.5 --nstdb-offset first_half(训练噪声偏移限前半), "
      "评测偏移限后半(last_half), 27 格与 clean; 见 runlog/W8/run_e6_noiseiso_chain.py")


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


def row_done(ckpt, seed, evalkind, ds, cond):
    if not OUT.exists():
        return False
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["ckpt"], r["seed"], r["eval"], r["downstream"], r.get("cond", "")) == \
                    (ckpt, str(seed), evalkind, ds, cond):
                return True
    return False


def record(ckpt, seed, evalkind, ds, cond, auroc, auprc, cksha):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "cond", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "protocol_id"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), ckpt, seed, evalkind, ds, cond,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha, "w8-noiseiso", HP])
    log(f"  + {ckpt} s{seed} {evalkind}/{ds}/{cond}: {auroc:.4f}/{auprc:.4f}")


def run_pt(seed):
    ck = CKD / f"seed{seed}"
    if (ck / "encoder_group.pth").exists() and (ck / "config.json").exists():
        log(f"noiseiso s{seed} PT: 已完成, 跳过")
        return True
    ck.mkdir(parents=True, exist_ok=True)
    with open_out(LOGD, f"noiseiso_pt_seed{seed}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_pt.py", "--data-dir", "data/pt_pretrain_nfh",
             "--epochs", "100", "--batch-size", "128", "--workers", "4",
             "--seed", str(seed), "--trc", "0",
             "--nstdb-aug", "0.5", "--nstdb-offset", "first_half",
             "--resume", "--checkpoint-dir", str(ck)],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    ok = rc == 0 and (ck / "encoder_group.pth").exists()
    log(f"noiseiso s{seed} PT: exit={rc} {'OK' if ok else 'FAIL(见 logs)'}")
    return ok


def run_lp(ck, seed, ds, nc):
    tag = f"noiseiso_{ds}_lp_seed{seed}"
    if row_done("noiseiso", seed, "lp", ds, "clean") and \
            (OUTD / "predictions" / tag / "y_prob.npy").exists():
        log(f"{tag}: 已有账+预测, 跳过")
        return True
    feat = OUTD / "feat" / tag
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--checkpoint", str(ck / "encoder_group.pth"), "--feat-dir", str(feat),
             "--seed", str(seed), "--workers", "6", "--trc", "0",
             "--extended-metrics", "1",
             "--save-predictions", f"runlog/W8/predictions/{tag}",
             "--protocol-id", "w8-noiseiso"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
        record("noiseiso", seed, "lp", ds, "clean", m["auroc"], m["auprc"],
               sha256_of(ck / "encoder_group.pth"))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def _noise_eval_seed(seed):
    """27 格后半段偏移评测: noiseiso vs C1(同噪声实现配对)。"""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from run_robustness import load_test, metrics_from_probs, _match_snr, _rng_for
    from run_robustness_nstdb import load_nstdb, build_lp
    from data_utils.nstdb_aug import draw_offset
    from models.vgg_1d import VGG16
    from models.linear import LinearClassifier
    from utils.checkpoint import load_torch_checkpoint
    dev = "cuda"

    class _GenWrap:
        """np.random.Generator 只有 integers(); 包一层给 draw_offset 的 randint 接口。"""
        def __init__(self, g):
            self._g = g
        def randint(self, lo, hi):
            return int(self._g.integers(lo, hi))

    def make_perturbed_lasthalf(x, ds, name, intensity):
        noise = load_nstdb(name.split("_")[1])
        Ln, L = noise.shape[1], x.shape[2]
        rng = _GenWrap(_rng_for(ds, name, intensity))
        offs = [draw_offset(rng, Ln, L, "last_half") for _ in range(x.shape[0])]
        ch = np.array([i % 2 for i in range(x.shape[1])])
        n = np.stack([noise[ch, o:o + L] for o in offs])
        return x + _match_snr(x, n, float(intensity))

    ck = CKD / f"seed{seed}"
    ckn = sha256_of(ck / "encoder_group.pth")
    cpt = load_torch_checkpoint(ck / "encoder_group.pth", map_location="cpu")
    encs = []
    for i in range(8):
        v = VGG16(ch_in=1, alpha=0.125, trc=0)
        sd = cpt["backbone_state_dict_list"][i] if "backbone_state_dict_list" in cpt \
            else cpt["backbone_state_dict"][i].state_dict()
        v.load_state_dict(sd, strict=False)
        encs.append(torch.nn.Sequential(*list(v.children())[:-1]).to(dev).eval())

    def feats(x_np, bs=256):
        x_t = torch.from_numpy(np.ascontiguousarray(x_np)).to(dev)
        out = []
        with torch.no_grad():
            for b0 in range(0, x_t.shape[0], bs):
                xb = x_t[b0:b0 + bs]
                out.append(torch.cat([encs[j](xb[:, [j], :]).squeeze(-1) for j in range(8)],
                                     dim=1).cpu())
        return torch.cat(out).to(dev)

    deltas = []
    for ds, nc in DOMAINS:
        head_n = __import__("models.linear", fromlist=["LinearClassifier"]).LinearClassifier(
            feat_dim=512, num_classes=nc).to(dev).eval()
        head_n.load_state_dict(torch.load(
            OUTD / f"feat/noiseiso_{ds}_lp_seed{seed}/classifier_best_ckpt.pth",
            map_location=dev, weights_only=True))
        encs_c1, head_c1, ckd1 = build_lp("c1", seed, ds, "w5" if ds == "cpsc" else "w2")
        ckc1 = sha256_of(ckd1 / "encoder_group.pth")
        x, y, _, _ = load_test(ds)
        for name in ("bw", "ma", "em"):
            for inten in ("0", "5", "10"):
                cond = f"{name}@{inten}"
                xp = make_perturbed_lasthalf(x, ds, "nstb_" + name, inten)
                with torch.no_grad():
                    pn = F.softmax(head_n(feats(xp)), dim=1).cpu().numpy()
                    xc_t = torch.from_numpy(np.ascontiguousarray(xp)).to(dev)
                    fc1 = torch.cat([encs_c1[j](xc_t[:, [j], :]).squeeze(-1) for j in range(8)],
                                    dim=1)
                    pc1 = F.softmax(head_c1(fc1), dim=1).cpu().numpy()
                au_n, ap_n = metrics_from_probs(y, pn, nc)
                au_c, ap_c = metrics_from_probs(y, pc1, nc)
                if not row_done("noiseiso", seed, "lpnoise", ds, cond):
                    record("noiseiso", seed, "lpnoise", ds, cond, au_n, ap_n, ckn)
                if not row_done("c1-lasteval", seed, "lpnoise", ds, cond):
                    record("c1-lasteval", seed, "lpnoise", ds, cond, au_c, ap_c, ckc1)
                deltas.append((au_n - au_c) * 100)
        del encs_c1, head_c1
        torch.cuda.empty_cache()
    return deltas


def summarize(seed, deltas):
    import numpy as np
    m = float(np.mean(deltas)) if deltas else float("nan")
    lines = [f"# E6 noiseiso seed{seed} 判定", "",
             f"- 27 格后半段偏移 ΔAUROC 均值(noiseiso − c1-lasteval, 同种子同格配对) = "
             f"{m:+.2f}pt",
             f"- 门(任务书): 增益方向与 W7 E5 原版(+2.17/+2.59/+2.59)一致 => "
             f"{'✅方向为正, 守住非片段记忆' if m > 0 else '❌方向为负/零, 片段记忆嫌疑, 如实上报'}",
             "- 幅度变化(与原版差距)如实报告, 不设幅度门"]
    (OUTD / f"e6_summary_seed{seed}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"e6_summary_seed{seed}.md written: mean={m:+.2f}pt")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["pt", "all", "eval"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    LOGD.mkdir(parents=True, exist_ok=True)
    (OUTD / "feat").mkdir(parents=True, exist_ok=True)
    (OUTD / "predictions").mkdir(parents=True, exist_ok=True)
    seeds = [args.seed] if args.seed is not None else SEEDS
    for seed in seeds:
        ck = CKD / f"seed{seed}"
        if args.phase in ("pt", "all"):
            log(f"=== noiseiso seed{seed}: PT 启动 ===")
            if not run_pt(seed):
                sys.exit(1)
        if args.phase == "all":
            for ds, nc in DOMAINS:
                log(f"noiseiso s{seed} lp/{ds}: 启动")
                if not run_lp(ck, seed, ds, nc):
                    sys.exit(1)
            need_eval = any(not row_done("noiseiso", seed, "lpnoise", ds, f"{n}@{i}")
                            for ds, _ in DOMAINS for n in ("bw", "ma", "em") for i in ("0", "5", "10"))
            if need_eval:
                deltas = _noise_eval_seed(seed)
                summarize(seed, deltas)
        elif args.phase == "eval":
            deltas = _noise_eval_seed(seed)
            summarize(seed, deltas)
    log("E6 链(本轮调度部分)完成")


if __name__ == "__main__":
    main()
