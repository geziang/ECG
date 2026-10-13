# -*- coding: utf-8 -*-
"""JOINT_20261013 联合验证链: M3/M4/A1/A2/M6 五配置 × seeds{0,2,4}。

协议入口: runlog/JOINT_20261013/protocol.md; 执行版任务书: 同目录/任务书.md。
每配置×种子 = PT(NFH 100ep) -> LP{ptbxl,cpsc,chapman} -> FT10(ptbxl)
-> [仅 M4/M6] 27 格噪声评测(last_half, 含同噪声配对参照行)。
全程串行(USB 盘 IO 纪律), 幂等: PT 看 encoder+config; LP/FT 看 csv 行+预测文件;
噪声格看 csv 行。失败即退出码 1, 由值班 cron 报警。

用法(conda DL 解释器, 仓库根为 cwd):
    python runlog/JOINT_20261013/run_joint_chain.py                 # 全部(默认含 M6)
    python runlog/JOINT_20261013/run_joint_chain.py --configs m3,m4
    python runlog/JOINT_20261013/run_joint_chain.py --configs m4 --phase noise --seed 0
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.pathguard import open_out

PY = sys.executable
OUTD = ROOT / "runlog/JOINT_20261013"
OUT = OUTD / "results.csv"
LOGD = OUTD / "logs"
MANIFEST = OUTD / "checkpoints_manifest.json"
SEEDS = [0, 2, 4]
DOMAINS = [("ptbxl", 5), ("cpsc", 9), ("chapman", 4)]
PROTO = "joint-20261013"

# (name, ckpt 目录名, trc, dv, rr, noise, rpeak 显式传参, hparams 描述, 噪声参照)
#   noise=None -> 无 nstdb; rpeak 显式传参仅 Stage4 分支(dv/rr>0)需要——A1/A2 补上
#   W8 E5 漏传的教训, 一律显式。noise_ref: m4->m3-lasteval(本链自有头), m6->m1-lasteval(c2 冻结头)。
CONFIGS = {
    "m3": dict(ckdir="joint_m3", trc=1, dv=0.01, rr=0.1, noise=None, rpeak=True,
               noise_ref=None,
               hp="JOINT M3: trc1+dv0.01+rr0.1, NFH 100ep, protocol §4.3 逐字"),
    "m4": dict(ckdir="joint_m4_noise", trc=1, dv=0.01, rr=0.1,
               noise=("0.5", "first_half"), rpeak=True, noise_ref="m3",
               hp="JOINT M4: trc1+dv0.01+rr0.1+nstdb0.5/first_half, NFH 100ep"),
    "a1": dict(ckdir="joint_a1", trc=0, dv=0.01, rr=0.0, noise=None, rpeak=True,
               noise_ref=None,
               hp="JOINT A1: dv0.01/rr0(Beat-only), 三种子重跑(W8 E5 s0 错缓存, 任务书 §二)"),
    "a2": dict(ckdir="joint_a2", trc=0, dv=0.0, rr=0.1, noise=None, rpeak=True,
               noise_ref=None,
               hp="JOINT A2: dv0/rr0.1(RR-only), 三种子重跑(同上)"),
    "m6": dict(ckdir="joint_m6", trc=1, dv=0.0, rr=0.0,
               noise=("0.5", "first_half"), rpeak=False, noise_ref="m1",
               hp="JOINT M6: trc1+nstdb0.5/first_half(噪声增益是否依赖 Beat-RR)"),
    # N1/N2 反事实(protocol §3.3, 用户 10-13 指令"最新需求一并做"): 先 s0, 必要时补 3 种子。
    # 依赖 apply_n1n2.py 落地的 --dv-rand-window/--rr-shuffle 开关(默认关==基线一致)。
    "n1": dict(ckdir="joint_n1", trc=1, dv=0.01, rr=0.1, noise=None, rpeak=True,
               noise_ref=None, seeds=[0], extra=["--dv-rand-window", "1"],
               hp="JOINT N1: M3+随机心拍窗口(反事实验证 R 峰对齐必要性), s0"),
    "n2": dict(ckdir="joint_n2", trc=1, dv=0.01, rr=0.1, noise=None, rpeak=True,
               noise_ref=None, seeds=[0], extra=["--rr-shuffle", "1"],
               hp="JOINT N2: M3+打乱 RR 标签(反事实验证节律目标信息量), s0"),
}
ORDER = ["m3", "m4", "a1", "a2", "m6", "n1", "n2"]


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def event(msg):
    with open_out(OUTD, "events.md", mode="a", encoding="utf-8") as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} {msg}\n")


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


def record(ckpt, seed, evalkind, ds, cond, auroc, auprc, cksha, hp):
    new = not OUT.exists()
    with open_out(OUTD, "results.csv", mode="a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "cond",
                        "auroc", "auprc", "git_sha", "checkpoint_sha256",
                        "protocol_id", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), ckpt, seed, evalkind, ds, cond,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha, PROTO, hp])
    log(f"  + {ckpt} s{seed} {evalkind}/{ds}/{cond}: {auroc:.4f}/{auprc:.4f}")


def manifest_update(name, seed, **kv):
    m = {}
    if MANIFEST.exists():
        try:
            m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            m = {}
    slot = m.setdefault(name, {}).setdefault(str(seed), {})
    slot.update(kv)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


class VRamPoller(threading.Thread):
    """30s 轮询 nvidia-smi 显存, 记录峰值(protocol §9)。失败静默(不影响主链)。

    注意属性名用 _halt: 不能叫 _stop —— 会遮蔽 threading.Thread 内部私有方法
    Thread._stop(), 线程收尾时内部以 self._stop() 调用之, Event 不可调用即崩
    (10-13 20:28 链首崩根因)。
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._halt = threading.Event()
        self.peak = 0

    def run(self):
        while not self._halt.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=20)
                self.peak = max(self.peak, int(r.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            self._halt.wait(30)

    def stop(self):
        self._halt.set()
        self.join(timeout=25)
        return self.peak


def ck_dir(name, seed):
    return ROOT / "checkpoint" / CONFIGS[name]["ckdir"] / f"seed{seed}"


def run_pt(name, seed):
    c = CONFIGS[name]
    ck = ck_dir(name, seed)
    if (ck / "encoder_group.pth").exists() and (ck / "config.json").exists():
        log(f"{name} s{seed} PT: 已完成, 跳过")
        return True
    ck.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", "run_pt.py", "--data-dir", "data/pt_pretrain_nfh",
           "--epochs", "100", "--batch-size", "128", "--workers", "4",
           "--seed", str(seed), "--trc", str(c["trc"])]
    if c["dv"] > 0 or c["rr"] > 0:
        cmd += ["--dualview-weight", str(c["dv"]), "--rr-weight", str(c["rr"])]
        if c["rpeak"]:
            cmd += ["--rpeak-npz", "data/pt_rpeaks_nfh.npz"]
    if c["noise"] is not None:
        cmd += ["--nstdb-aug", c["noise"][0], "--nstdb-offset", c["noise"][1]]
    cmd += c.get("extra", [])
    cmd += ["--resume", "--checkpoint-dir", str(ck)]
    log(f"{name} s{seed} PT 启动: {' '.join(cmd[2:])}")
    poll = VRamPoller()
    poll.start()
    t0 = time.time()
    with open_out(LOGD, f"{name}_pt_seed{seed}.log", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            cwd=str(ROOT)).returncode
    wall = time.time() - t0
    peak = poll.stop()
    ok = rc == 0 and (ck / "encoder_group.pth").exists()
    manifest_update(name, seed, pt_wall_s=round(wall, 1), vram_peak_mib=peak,
                    git_sha=git_sha(), ckpt_sha256=sha256_of(ck / "encoder_group.pth")
                    if ok else None, done=ok)
    log(f"{name} s{seed} PT: exit={rc} wall={wall/60:.0f}min vram_peak={peak}MiB "
        f"{'OK' if ok else 'FAIL(见 logs)'}")
    if ok:
        event(f"{name} seed{seed} PT 完成: {wall/60:.0f}min, 峰值显存 {peak}MiB")
    return ok


def run_lp(name, seed, ds, nc):
    c = CONFIGS[name]
    ck = ck_dir(name, seed)
    tag = f"{name}_{ds}_lp_seed{seed}"
    if row_done(name, seed, "lp", ds, "clean") and \
            (OUTD / "predictions" / tag / "y_prob.npy").exists():
        log(f"{tag}: 已有账+预测, 跳过")
        return True
    feat = OUTD / "feat" / tag
    t0 = time.time()
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--checkpoint", str(ck / "encoder_group.pth"), "--feat-dir", str(feat),
             "--seed", str(seed), "--workers", "6", "--trc", str(c["trc"]),
             "--extended-metrics", "1",
             "--save-predictions", f"runlog/JOINT_20261013/predictions/{tag}",
             "--protocol-id", PROTO],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
        record(name, seed, "lp", ds, "clean", m["auroc"], m["auprc"],
               sha256_of(ck / "encoder_group.pth"), c["hp"])
        manifest_update(name, seed, **{f"lp_{ds}_wall_s": round(time.time() - t0, 1)})
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def run_ft10(name, seed):
    c = CONFIGS[name]
    ck = ck_dir(name, seed)
    tag = f"{name}_ptbxl_ft10_seed{seed}"
    if row_done(name, seed, "ft10", "ptbxl", "clean") and \
            (OUTD / "ft" / tag / "metrics.json").exists():
        log(f"{tag}: 已有账, 跳过")
        return True
    mdir = OUTD / "ft" / tag
    t0 = time.time()
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_ft.py", "--data-dir", "data/ptbxl", "--num-classes", "5",
             "--fraction", "0.1", "--checkpoint", str(ck / "encoder_group.pth"),
             "--model-dir", str(mdir), "--workers", "4", "--epochs", "100",
             "--batch-size", "128", "--learning-rate", "0.0001",
             "--seed", str(seed), "--trc", str(c["trc"]),
             "--extended-metrics", "1",
             "--save-predictions", f"runlog/JOINT_20261013/predictions/{tag}",
             "--protocol-id", PROTO],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
        record(name, seed, "ft10", "ptbxl", "clean", m["auroc"], m["auprc"],
               sha256_of(ck / "encoder_group.pth"), c["hp"])
        manifest_update(name, seed, ft10_wall_s=round(time.time() - t0, 1))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def load_encoders(ckpt_path, trc):
    """逐导联编码器(去 fc), 兼容双格式 checkpoint。"""
    import torch
    from models.vgg_1d import VGG16
    from utils.checkpoint import load_torch_checkpoint
    cpt = load_torch_checkpoint(ckpt_path, map_location="cpu")
    encs = []
    for i in range(8):
        v = VGG16(ch_in=1, alpha=0.125, trc=trc)
        sd = cpt["backbone_state_dict_list"][i] if "backbone_state_dict_list" in cpt \
            else cpt["backbone_state_dict"][i].state_dict()
        v.load_state_dict(sd, strict=False)
        encs.append(torch.nn.Sequential(*list(v.children())[:-1]).to("cuda").eval())
    return encs


def noise_eval(name, seed):
    """27 格 last_half 评测(沿用 W8 E6 实现): 本配置 + 同噪声配对参照各记行。

    m4 -> 参照 m3-lasteval(本链 m3 encoder+LP 头, trc1);
    m6 -> 参照 m1-lasteval(c2 冻结锚点, build_lp)。
    噪声实现确定性: _rng_for(ds,name,intensity) 同 ds/类型/强度同噪声段——与
    W8 E6 的 noiseiso/c1-lasteval 行同口径可直接配对。
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from run_robustness import load_test, metrics_from_probs, _match_snr, _rng_for
    from run_robustness_nstdb import load_nstdb, build_lp
    from data_utils.nstdb_aug import draw_offset
    from models.linear import LinearClassifier
    c = CONFIGS[name]
    dev = "cuda"

    class _GenWrap:
        def __init__(self, g):
            self._g = g

        def randint(self, lo, hi):
            return int(self._g.integers(lo, hi))

    def make_perturbed_lasthalf(x, ds, nname, intensity):
        noise = load_nstdb(nname.split("_")[1])
        Ln, L = noise.shape[1], x.shape[2]
        rng = _GenWrap(_rng_for(ds, nname, intensity))
        offs = [draw_offset(rng, Ln, L, "last_half") for _ in range(x.shape[0])]
        ch = np.array([i % 2 for i in range(x.shape[1])])
        n = np.stack([noise[ch, o:o + L] for o in offs])
        return x + _match_snr(x, n, float(intensity))

    ck = ck_dir(name, seed)
    ckn = sha256_of(ck / "encoder_group.pth")
    encs = load_encoders(ck / "encoder_group.pth", c["trc"])

    def feats(enc_list, x_np, bs=256):
        x_t = torch.from_numpy(np.ascontiguousarray(x_np)).to(dev)
        out = []
        with torch.no_grad():
            for b0 in range(0, x_t.shape[0], bs):
                xb = x_t[b0:b0 + bs]
                out.append(torch.cat(
                    [enc_list[j](xb[:, [j], :]).squeeze(-1) for j in range(8)], dim=1)
                    .cpu())
        return torch.cat(out).to(dev)

    t0 = time.time()
    for ds, nc in DOMAINS:
        head = LinearClassifier(feat_dim=512, num_classes=nc).to(dev).eval()
        head.load_state_dict(torch.load(
            OUTD / f"feat/{name}_{ds}_lp_seed{seed}/classifier_best_ckpt.pth",
            map_location=dev, weights_only=True))
        if c["noise_ref"] == "m3":
            ck3 = ck_dir("m3", seed)
            encs_r = load_encoders(ck3 / "encoder_group.pth", 1)
            head_r = LinearClassifier(feat_dim=512, num_classes=nc).to(dev).eval()
            head_r.load_state_dict(torch.load(
                OUTD / f"feat/m3_{ds}_lp_seed{seed}/classifier_best_ckpt.pth",
                map_location=dev, weights_only=True))
            cksha_r, ref_tag = sha256_of(ck3 / "encoder_group.pth"), "m3-lasteval"
        else:  # m1
            encs_r, head_r, ckd1 = build_lp("c2", seed, ds, "w5" if ds == "cpsc" else "w2")
            cksha_r, ref_tag = sha256_of(ckd1 / "encoder_group.pth"), "m1-lasteval"
        x, y, _, _ = load_test(ds)
        for nname in ("bw", "ma", "em"):
            for inten in ("0", "5", "10"):
                cond = f"{nname}@{inten}"
                probs_ok = all(
                    (OUTD / "predictions" / f"{t}_{ds}_lpnoise_seed{seed}" / cond
                     / "y_prob.npy").exists() for t in (name, "m3" if c["noise_ref"] == "m3" else "m1"))
                if row_done(name, seed, "lpnoise", ds, cond) and \
                        row_done("m3" if c["noise_ref"] == "m3" else "m1", seed,
                                 "lpnoise", ds, cond) and probs_ok:
                    continue
                xp = make_perturbed_lasthalf(x, ds, "nstb_" + nname, inten)
                with torch.no_grad():
                    pn = F.softmax(head(feats(encs, xp)), dim=1).cpu().numpy()
                    pr = F.softmax(head_r(feats(encs_r, xp)), dim=1).cpu().numpy()
                au, ap = metrics_from_probs(y, pn, nc)
                aur, apr = metrics_from_probs(y, pr, nc)
                if not row_done(name, seed, "lpnoise", ds, cond):
                    record(name, seed, "lpnoise", ds, cond, au, ap, ckn, c["hp"])
                if not row_done(ref_tag, seed, "lpnoise", ds, cond):
                    record(ref_tag, seed, "lpnoise", ds, cond, aur, apr, cksha_r,
                           "JOINT 参照行: 同噪声实现配对(last_half), 见任务书 §四")
                # protocol §9: 逐记录预测+噪声片段标识(确定性重放可复现同噪声)
                for tag_, p_ in ((name, pn), (ref_tag, pr)):
                    pdir = OUTD / "predictions" / f"{tag_}_{ds}_lpnoise_seed{seed}" / cond
                    if not (pdir / "y_prob.npy").exists():
                        pdir.mkdir(parents=True, exist_ok=True)
                        np.save(pdir / "y_prob.npy", p_)
                        np.save(pdir / "y_true.npy", y)
                        np.save(pdir / "noise_offsets.npy", np.asarray(offs))
        del encs_r, head_r, head
        torch.cuda.empty_cache()
    manifest_update(name, seed, noise27_wall_s=round(time.time() - t0, 1))
    log(f"{name} s{seed} 27 格噪声评测完成({(time.time()-t0)/60:.0f}min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(ORDER),
                    help="逗号分隔, 顺序敏感; 默认 m3,m4,a1,a2,m6")
    ap.add_argument("--phase", default="all", choices=["pt", "lp", "ft", "noise", "all"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    names = [s.strip() for s in args.configs.split(",") if s.strip()]
    for n in names:
        assert n in CONFIGS, n
    for sub in ("feat", "predictions", "ft", "logs"):
        (OUTD / sub).mkdir(parents=True, exist_ok=True)
    seeds = [args.seed] if args.seed is not None else SEEDS
    log(f"JOINT 链启动: configs={names} phase={args.phase} seeds={seeds} 解释器={PY}")
    for n in names:
        for seed in ([args.seed] if args.seed is not None
                     else CONFIGS[n].get("seeds", SEEDS)):
            if args.phase in ("pt", "all"):
                if not run_pt(n, seed):
                    event(f"{n} seed{seed} PT 失败, 链终止(logs/{n}_pt_seed{seed}.log)")
                    sys.exit(1)
            if args.phase in ("lp", "all"):
                for ds, nc in DOMAINS:
                    if not run_lp(n, seed, ds, nc):
                        sys.exit(1)
            if args.phase in ("ft", "all") and not run_ft10(n, seed):
                sys.exit(1)
            if args.phase in ("noise", "all") and CONFIGS[n]["noise_ref"]:
                noise_eval(n, seed)
        event(f"{n} seeds{seeds} phase={args.phase} 段完成")
    log("JOINT 链(本轮调度部分)完成")
    event("JOINT 链全部完成" if args.phase == "all" else f"JOINT 链 {names} phase={args.phase} 完成")


if __name__ == "__main__":
    main()
