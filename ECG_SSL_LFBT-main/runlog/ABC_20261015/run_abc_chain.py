# -*- coding: utf-8 -*-
"""ABC_20261015 联合框架补充实验主链: P0+P1 一次性连续队列(方案 ABC-SUPP-20261015)。

协议入口: runlog/ABC_20261015/protocol_frozen.md(冻结版); 方案原件同目录。
阶段序: audit → smoke → ft10fill(P0-1,18格) → m7 → m8 → m9 → bench(P0-4) →
        stats0(P0统计快照) → n1n2(P1-1,s2/s4) → m4b(P1-2,第二语料) →
        multilib(P1-3) → stats(全量) → report
全程串行(F 盘 USB IO 纪律), 幂等: PT 看 encoder+config; LP/FT 看账行+预测文件;
噪声格看账行。失败 exit 1, 值班 cron 报警 + launch_abc.ps1 幂等重启续跑。

用法(conda DL 解释器, 仓库根为 cwd):
    python runlog/ABC_20261015/run_abc_chain.py                    # 全阶段
    python runlog/ABC_20261015/run_abc_chain.py --stages audit     # 单阶段
    python runlog/ABC_20261015/run_abc_chain.py --stages ft10ext   # 可选: M2/M5 补FT10(默认不跑)
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
OUTD = ROOT / "runlog/ABC_20261015"
OUT = OUTD / "results.csv"
LOGD = OUTD / "logs"
MANIFEST = OUTD / "checkpoints_manifest.json"
SEEDS = [0, 2, 4]
DOMAINS = [("ptbxl", 5), ("cpsc", 9), ("chapman", 4)]
PROTO = "abc-20261015"
CORPORA = {  # 语料 -> (data-dir, rpeak-npz)
    "nfh": ("data/pt_pretrain_nfh", "data/pt_rpeaks_nfh.npz"),
    "ptbxl": ("data/pt_pretrain", "data/pt_rpeaks.npz"),
}

# (名称, ckpt 目录, trc, dv, rr, noise, rpeak 显式传参, 语料, 下游计划)
#   下游计划: lp=LP 域列表 / ft=FT10 域列表 / noise=27|None / noise9=(仅 m4b, cpsc 9 格)
#   m3/m4/m6 只为 ft10fill 引用(PT 自动跳过); 其 LP/噪声行在 JOINT 账本, 不重跑。
CONFIGS = {
    "m3": dict(ckdir="joint_m3", trc=1, dv=0.01, rr=0.1, noise=None, rpeak=True,
               corpus="nfh", lp=[], ft=[], noise27=False,
               hp="JOINT M3 锚点(本链只补 cpsc/chapman FT10)"),
    "m4": dict(ckdir="joint_m4_noise", trc=1, dv=0.01, rr=0.1,
               noise=("0.5", "first_half"), rpeak=True, corpus="nfh",
               lp=[], ft=[], noise27=False,
               hp="JOINT M4 锚点(本链只补 cpsc/chapman FT10)"),
    "m6": dict(ckdir="joint_m6", trc=1, dv=0.0, rr=0.0,
               noise=("0.5", "first_half"), rpeak=False, corpus="nfh",
               lp=[], ft=[], noise27=False,
               hp="JOINT M6 锚点(本链只补 cpsc/chapman FT10)"),
    "m7": dict(ckdir="abc_supp/m7_bc", trc=0, dv=0.01, rr=0.1,
               noise=("0.5", "first_half"), rpeak=True, corpus="nfh",
               lp=[d for d, _ in DOMAINS], ft=[d for d, _ in DOMAINS], noise27=True,
               hp="ABC M7: trc0+dv0.01+rr0.1+nstdb0.5/first_half (B+C 闭合格)"),
    "m8": dict(ckdir="abc_supp/m8_a_beat_c", trc=1, dv=0.01, rr=0.0,
               noise=("0.5", "first_half"), rpeak=True, corpus="nfh",
               lp=[d for d, _ in DOMAINS], ft=[d for d, _ in DOMAINS], noise27=True,
               hp="ABC M8: trc1+dv0.01/rr0+nstdb0.5/first_half (A+Beat+C)"),
    "m9": dict(ckdir="abc_supp/m9_a_rr_c", trc=1, dv=0.0, rr=0.1,
               noise=("0.5", "first_half"), rpeak=True, corpus="nfh",
               lp=[d for d, _ in DOMAINS], ft=[d for d, _ in DOMAINS], noise27=True,
               hp="ABC M9: trc1+dv0/rr0.1+nstdb0.5/first_half (A+RR+C)"),
    # P1-1: N1/N2 补 s2/s4(s0=JOINT 已有); 下游=三域 LP + PTB-XL FT10(方案推荐版)
    "n1": dict(ckdir="joint_n1", trc=1, dv=0.01, rr=0.1, noise=None, rpeak=True,
               corpus="nfh", seeds=[2, 4], lp=[d for d, _ in DOMAINS],
               ft=["ptbxl"], noise27=False, extra=["--dv-rand-window", "1"],
               hp="ABC N1: M3+随机心拍窗口, 补 s2/s4(机制结论升三种子)"),
    "n2": dict(ckdir="joint_n2", trc=1, dv=0.01, rr=0.1, noise=None, rpeak=True,
               corpus="nfh", seeds=[2, 4], lp=[d for d, _ in DOMAINS],
               ft=["ptbxl"], noise27=False, extra=["--rr-shuffle", "1"],
               hp="ABC N2: M3+打乱 RR 标签, 补 s2/s4"),
    # P1-2: 第二语料(PTB-XL 预训练)完整 M4; R 峰缓存=pt_rpeaks.npz(禁复用 NFH)
    "m4b": dict(ckdir="abc_supp/m4_ptbxl", trc=1, dv=0.01, rr=0.1,
                noise=("0.5", "first_half"), rpeak=True, corpus="ptbxl",
                lp=[d for d, _ in DOMAINS], ft=["cpsc"], noise27=False, noise9=True,
                hp="ABC M4b: M4 于第二语料 data/pt_pretrain + pt_rpeaks.npz"),
    "m2": dict(ckdir="w6_dv", trc=0, dv=0.01, rr=0.1, noise=None, rpeak=True,
               corpus="nfh", ft=[], hp="M2 锚点(ft10ext 可选用)"),
    "m5": dict(ckdir="w8_noiseiso", trc=0, dv=0.0, rr=0.0,
               noise=("0.5", "first_half"), rpeak=False, corpus="nfh", ft=[],
               hp="M5 锚点(ft10ext 可选用)"),
}
# M2 锚点跨批次目录映射
def m2_ckdir(seed):
    return ROOT / ("checkpoint/w6_dv/seed0" if seed == 0
                   else f"checkpoint/w7_dv/seed{seed}")

ANCHORS = {  # audit 用: 矩阵编号 -> [(ckpt 路径, seed)]（m2 特例单独处理）
    "m0": [("checkpoint/confirm/c1_seed{}", s) for s in SEEDS],
    "m1": [("checkpoint/confirm/c2_seed{}", s) for s in SEEDS],
    "m3": [("checkpoint/joint_m3/seed{}", s) for s in SEEDS],
    "m4": [("checkpoint/joint_m4_noise/seed{}", s) for s in SEEDS],
    "m5": [("checkpoint/w8_noiseiso/seed{}", s) for s in SEEDS],
    "m6": [("checkpoint/joint_m6/seed{}", s) for s in SEEDS],
    "n1_s0": [("checkpoint/joint_n1/seed0", 0)],
    "n2_s0": [("checkpoint/joint_n2/seed0", 0)],
}
STAGES = ["audit", "smoke", "ft10fill", "m7", "m8", "m9", "bench",
          "stats0", "n1n2", "m4b", "multilib", "stats", "report"]


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


# ---------------------------------------------------------------- 账本(§14 schema)
HEADER = ["ts", "model", "seed", "pretrain_source", "task", "dataset",
          "label_fraction", "noise_type", "snr", "auroc", "auprc", "git_sha",
          "checkpoint_sha", "split_sha", "protocol_id", "hparams_ref"]


def row_done(model, seed, task, ds, ntype, snr):
    if not OUT.exists():
        return False
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["model"], r["seed"], r["task"], r["dataset"],
                    r["noise_type"], r["snr"]) == (model, str(seed), task, ds, ntype, snr):
                return True
    return False


def record(model, seed, task, ds, ntype, snr, auroc, auprc, cksha, hp,
           fraction="", split_sha=""):
    new = not OUT.exists()
    with open_out(OUTD, "results.csv", mode="a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), model, seed,
                    CONFIGS[model]["corpus"] if model in CONFIGS else
                    ("nfh" if model in ("m0", "m1", "m2", "m3", "m5", "m6") else ""),
                    task, ds, fraction, ntype, snr,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha,
                    split_sha or split_sha_of(ds), PROTO, hp])
    log(f"  + {model} s{seed} {task}/{ds}/{ntype}@{snr}: {auroc:.4f}/{auprc:.4f}")


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


# ---------------------------------------------------------------- split sha 缓存
_SPLIT_CACHE = {}


def split_sha_of(ds):
    """数据集 train/val/test 排序文件清单 SHA256(类目录序+文件名序, 与加载器一致口径)。"""
    if ds in _SPLIT_CACHE:
        return _SPLIT_CACHE[ds]
    import os
    lines = []
    for split in ("train", "val", "test"):
        root = ROOT / "data" / ds / split
        lines.append(f"[{split}]")
        for cls in sorted(d for d in os.listdir(root) if (root / d).is_dir()):
            for fn in sorted(os.listdir(root / cls)):
                if fn.endswith(".npy"):
                    lines.append(f"{cls}/{fn}")
    payload = "\n".join(lines).encode("utf-8")
    sha = hashlib.sha256(payload).hexdigest()[:16]
    _SPLIT_CACHE[ds] = sha
    cache_f = OUTD / "data_sha.json"
    c = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {}
    if c.get(ds) != sha:
        c[ds] = sha
        cache_f.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
    # 样本 ID 清单(三目录排序顺序, 一次性落盘供追溯)
    order_f = OUTD / "predictions" / f"{ds}_sample_order.txt"
    if not order_f.exists():
        order_f.parent.mkdir(parents=True, exist_ok=True)
        order_f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha


class VRamPoller(threading.Thread):
    """30s 轮询 nvidia-smi 显存峰值。属性名 _halt(勿用 _stop, JOINT 10-13 教训)。"""

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
    if name == "m2":
        return m2_ckdir(seed)
    return ROOT / "checkpoint" / CONFIGS[name]["ckdir"] / f"seed{seed}"


# ---------------------------------------------------------------- PT / LP / FT
def pt_cmd(name, seed, epochs):
    c = CONFIGS[name]
    data_dir, rpeak = CORPORA[c["corpus"]]
    cmd = [PY, "-u", "run_pt.py", "--data-dir", data_dir,
           "--epochs", str(epochs), "--batch-size", "128", "--workers", "4",
           "--seed", str(seed), "--trc", str(c["trc"])]
    if c["dv"] > 0 or c["rr"] > 0:
        cmd += ["--dualview-weight", str(c["dv"]), "--rr-weight", str(c["rr"])]
        if c["rpeak"]:
            cmd += ["--rpeak-npz", rpeak]
    if c["noise"] is not None:
        cmd += ["--nstdb-aug", c["noise"][0], "--nstdb-offset", c["noise"][1]]
    cmd += c.get("extra", [])
    return cmd


def run_pt(name, seed, epochs=100):
    c = CONFIGS[name]
    ck = ck_dir(name, seed)
    if (ck / "encoder_group.pth").exists() and (ck / "config.json").exists():
        log(f"{name} s{seed} PT: 已完成, 跳过")
        return True
    ck.mkdir(parents=True, exist_ok=True)
    cmd = pt_cmd(name, seed, epochs) + ["--resume", "--checkpoint-dir", str(ck)]
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
                    pt_epochs=epochs, git_sha=git_sha(),
                    ckpt_sha256=sha256_of(ck / "encoder_group.pth") if ok else None,
                    done=ok)
    log(f"{name} s{seed} PT: exit={rc} wall={wall/60:.0f}min vram_peak={peak}MiB "
        f"{'OK' if ok else 'FAIL(见 logs)'}")
    if ok:
        event(f"{name} seed{seed} PT 完成: {wall/60:.0f}min, 峰值显存 {peak}MiB")
    return ok


def run_lp(name, seed, ds, nc):
    c = CONFIGS[name]
    ck = ck_dir(name, seed)
    tag = f"{name}_{ds}_lp_seed{seed}"
    if row_done(name, seed, "lp", ds, "", "") and \
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
             "--save-predictions", f"runlog/ABC_20261015/predictions/{tag}",
             "--protocol-id", PROTO],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
        record(name, seed, "lp", ds, "", "", m["auroc"], m["auprc"],
               sha256_of(ck / "encoder_group.pth"), c["hp"], fraction="1.0")
        manifest_update(name, seed, **{f"lp_{ds}_wall_s": round(time.time() - t0, 1)})
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        event(f"{tag} LP 失败 rc={rc}({type(e).__name__}: {e})")
        return False


def run_ft10(name, seed, ds, nc):
    """W2 FT10 冻结模板(=W6 ft10grid 逐字): fraction 0.1 / 100ep / bs128 / lr1e-4。"""
    c = CONFIGS[name]
    ck = ck_dir(name, seed)
    tag = f"{name}_{ds}_ft10_seed{seed}"
    if row_done(name, seed, "ft10", ds, "", "") and \
            (OUTD / "ft" / tag / "metrics.json").exists():
        log(f"{tag}: 已有账, 跳过")
        return True
    mdir = OUTD / "ft" / tag
    t0 = time.time()
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_ft.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--fraction", "0.1", "--checkpoint", str(ck / "encoder_group.pth"),
             "--model-dir", str(mdir), "--workers", "4", "--epochs", "100",
             "--batch-size", "128", "--learning-rate", "0.0001",
             "--seed", str(seed), "--trc", str(c["trc"]),
             "--extended-metrics", "1",
             "--save-predictions", f"runlog/ABC_20261015/predictions/{tag}",
             "--protocol-id", PROTO],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
        record(name, seed, "ft10", ds, "", "", m["auroc"], m["auprc"],
               sha256_of(ck / "encoder_group.pth"), c["hp"], fraction="0.1")
        manifest_update(name, seed, **{f"ft10_{ds}_wall_s": round(time.time() - t0, 1)})
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        event(f"{tag} FT10 失败 rc={rc}({type(e).__name__}: {e})")
        return False


# ---------------------------------------------------------------- 噪声评测(27/9 格)
def load_encoders(ckpt_path, trc):
    """逐导联编码器(去 fc), 兼容双格式 checkpoint; strict 校验 missing 仅 fc.*。"""
    import torch
    from models.vgg_1d import VGG16
    from utils.checkpoint import load_torch_checkpoint
    cpt = load_torch_checkpoint(ckpt_path, map_location="cpu")
    encs = []
    for i in range(8):
        v = VGG16(ch_in=1, alpha=0.125, trc=trc)
        sd = cpt["backbone_state_dict_list"][i] if "backbone_state_dict_list" in cpt \
            else cpt["backbone_state_dict"][i].state_dict()
        missing, unexpected = v.load_state_dict(sd, strict=False)
        assert missing == ["fc.weight", "fc.bias"] and unexpected == [], \
            (ckpt_path, missing, unexpected)
        encs.append(torch.nn.Sequential(*list(v.children())[:-1]).to("cuda").eval())
    return encs


def noise_eval(name, seed, cells="27"):
    """last_half 噪声评测, 无参照行(逐记录预测落盘, 统计阶段与 JOINT 行直接配对)。
    噪声实现确定性: _rng_for 同 ds/类型/强度同噪声段——与 JOINT/W8 E6 同口径。
    cells='27' 三域×3噪声×3SNR; '9' 仅 cpsc×3噪声×3SNR(M4b)。
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from run_robustness import load_test, metrics_from_probs, _match_snr, _rng_for
    from run_robustness_nstdb import load_nstdb
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
        return x + _match_snr(x, n, float(intensity)), np.asarray(offs)

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
    doms = DOMAINS if cells == "27" else [("cpsc", 9)]
    for ds, nc in doms:
        head = LinearClassifier(feat_dim=512, num_classes=nc).to(dev).eval()
        head.load_state_dict(torch.load(
            OUTD / f"feat/{name}_{ds}_lp_seed{seed}/classifier_best_ckpt.pth",
            map_location=dev, weights_only=True))
        x, y, _, _ = load_test(ds)
        ssha = split_sha_of(ds)
        for nname in ("bw", "ma", "em"):
            for inten in ("0", "5", "10"):
                cond = f"{nname}@{inten}"
                pdir = OUTD / "predictions" / f"{name}_{ds}_lpnoise_seed{seed}" / cond
                if row_done(name, seed, "lpnoise", ds, nname, inten) and \
                        (pdir / "y_prob.npy").exists():
                    continue
                xp, offs = make_perturbed_lasthalf(x, ds, "nstb_" + nname, inten)
                with torch.no_grad():
                    pn = F.softmax(head(feats(encs, xp)), dim=1).cpu().numpy()
                au, ap = metrics_from_probs(y, pn, nc)
                record(name, seed, "lpnoise", ds, nname, inten, au, ap, ckn,
                       c["hp"], fraction="1.0", split_sha=ssha)
                pdir.mkdir(parents=True, exist_ok=True)
                np.save(pdir / "y_prob.npy", pn)
                np.save(pdir / "y_true.npy", y)
                np.save(pdir / "noise_offsets.npy", np.asarray(offs))
        del head
        torch.cuda.empty_cache()
    manifest_update(name, seed,
                    **{f"noise{cells}_wall_s": round(time.time() - t0, 1)})
    log(f"{name} s{seed} {cells} 格噪声评测完成({(time.time()-t0)/60:.0f}min)")


# ---------------------------------------------------------------- 阶段: audit
def stage_audit():
    import os
    log("audit: 锚点 checkpoint / 语料缓存 / 数据分区核验")
    missing = []
    for label, entries in ANCHORS.items():
        for tpl, seed in entries:
            p = ROOT / tpl.format(seed)
            if not (p / "encoder_group.pth").exists():
                missing.append(str(p))
    for s in SEEDS:  # M2 跨批次
        if not (m2_ckdir(s) / "encoder_group.pth").exists():
            missing.append(str(m2_ckdir(s)))
    if missing:
        event("audit 失败: 锚点缺失 " + "; ".join(missing))
        sys.exit(f"[audit] 锚点 checkpoint 缺失: {missing}")
    for ds, _ in DOMAINS:
        for split in ("train", "val", "test"):
            if not (ROOT / "data" / ds / split).is_dir():
                sys.exit(f"[audit] 数据分区缺失: data/{ds}/{split}")
        split_sha_of(ds)  # 预计算并缓存 + 测试顺序样本清单落盘
    # R 峰缓存哈希 + 第二语料一一对应断言(方案 P1-2 硬门)
    import numpy as np
    rp_hash = {}
    for tag, (_, rp) in CORPORA.items():
        p = ROOT / rp
        if not p.exists():
            sys.exit(f"[audit] R 峰缓存缺失: {rp}")
        rp_hash[tag] = sha256_of(p)[:16]
    ids_pt = {f[:-4] for f in os.listdir(ROOT / "data/pt_pretrain/samples")
              if f.endswith(".npy")}
    keys_pt = set(np.load(ROOT / "data/pt_rpeaks.npz", allow_pickle=True).keys())
    ids_nfh = {f[:-4] for f in os.listdir(ROOT / "data/pt_pretrain_nfh/samples")
               if f.endswith(".npy")}
    keys_nfh = set(np.load(ROOT / "data/pt_rpeaks_nfh.npz", allow_pickle=True).keys())
    assert ids_pt == keys_pt, \
        f"[audit] pt_pretrain({len(ids_pt)}) 与 pt_rpeaks.npz({len(keys_pt)}) ID 不一一对应"
    assert ids_nfh == keys_nfh, \
        f"[audit] pt_pretrain_nfh({len(ids_nfh)}) 与 pt_rpeaks_nfh.npz({len(keys_nfh)}) ID 不一一对应"
    # 锚点 SHA 入 manifest(只记一次, 已有则跳过)
    for label, entries in ANCHORS.items():
        for tpl, seed in entries:
            ck = ROOT / tpl.format(seed)
            slot_key = "anchor_" + label.split("_")[0]
            man = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
            cur = man.get(slot_key, {}).get(str(seed), {})
            if "ckpt_sha256" not in cur:
                manifest_update(slot_key, seed, ckpt_sha256=sha256_of(ck / "encoder_group.pth"),
                                ckpt_dir=str(ck), audited_at=time.strftime("%Y-%m-%d %H:%M"))
    for s in SEEDS:
        man = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
        if "ckpt_sha256" not in man.get("anchor_m2", {}).get(str(s), {}):
            manifest_update("anchor_m2", s, ckpt_sha256=sha256_of(m2_ckdir(s) / "encoder_group.pth"),
                            ckpt_dir=str(m2_ckdir(s)), audited_at=time.strftime("%Y-%m-%d %H:%M"))
    event(f"audit 通过: 锚点齐(含 M2 跨批次), R峰缓存哈希 {rp_hash}, "
          f"pt_pretrain({len(ids_pt)})==pt_rpeaks.npz 严格一一对应, split_sha 就绪")
    log("audit OK")


# ---------------------------------------------------------------- 阶段: smoke(§9.1)
def _parse_loss_terms(log_path):
    terms, losses = set(), []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("{") and '"loss"' in line:
            try:
                d = json.loads(line)
            except Exception:
                continue
            losses.append(d)
            terms |= {k for k in d if k.startswith("loss")}
    return terms, losses


def stage_smoke():
    import shutil
    names = ["m7", "m8", "m9", "m4b"]
    smoke_root = ROOT / "checkpoint/abc_smoke"
    for name in names:
        marker = OUTD / "logs" / f"smoke_ok_{name}"
        if marker.exists():
            log(f"smoke {name}: 已通过, 跳过")
            continue
        c = CONFIGS[name]
        ck = smoke_root / name
        if ck.exists():
            shutil.rmtree(ck)
        ck.mkdir(parents=True, exist_ok=True)
        data_dir, rpeak = CORPORA[c["corpus"]]
        log(f"smoke {name}: 1 epoch 试跑 -> {ck}")
        cmd = pt_cmd(name, seed=0, epochs=1) + ["--checkpoint-dir", str(ck)]
        with open_out(LOGD, f"smoke_{name}.log", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(ROOT)).returncode
        lp = LOGD / f"smoke_{name}.log"
        terms, losses = _parse_loss_terms(lp)
        assert rc == 0 and losses, f"smoke {name}: PT exit={rc} 或无损失行"
        last = losses[-1]
        for k, v in last.items():
            if k.startswith("loss"):
                assert isinstance(v, (int, float)) and v == v and abs(v) < 1e9, \
                    f"smoke {name}: {k}={v} 非有限"
        assert ("loss_dv" in terms) == (c["dv"] > 0), f"smoke {name}: loss_dv 出现性与 dv 开关不符"
        assert ("loss_rr" in terms) == (c["rr"] > 0), f"smoke {name}: loss_rr 出现性与 rr 开关不符"
        cfg = json.loads((ck / "config.json").read_text(encoding="utf-8"))["args"]
        assert int(cfg["trc"]) == c["trc"] and abs(float(cfg["dualview_weight"]) - c["dv"]) < 1e-12 \
            and abs(float(cfg["rr_weight"]) - c["rr"]) < 1e-12, f"smoke {name}: config 开关不符 {cfg}"
        if c["noise"] is not None:
            assert abs(float(cfg["nstdb_aug"]) - 0.5) < 1e-12 \
                and cfg["nstdb_offset"] == "first_half", f"smoke {name}: 噪声开关不符"
        if c["rpeak"]:
            assert cfg["rpeak_npz"].replace("\\", "/").endswith(rpeak.split("/")[-1]), \
                f"smoke {name}: R 峰缓存不符({cfg['rpeak_npz']})——M4b 严禁复用 NFH 缓存"
        assert cfg["data_dir"].replace("\\", "/").endswith(data_dir.split("/")[-1]), \
            f"smoke {name}: 语料目录不符({cfg['data_dir']})"
        # resume 连续性: 2 epoch 续训
        cmd2 = pt_cmd(name, seed=0, epochs=2) + ["--resume", "--checkpoint-dir", str(ck)]
        with open_out(LOGD, f"smoke_{name}_resume.log", encoding="utf-8") as f:
            rc2 = subprocess.run(cmd2, stdout=f, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT)).returncode
        lp2 = LOGD / f"smoke_{name}_resume.log"
        txt2 = lp2.read_text(encoding="utf-8", errors="ignore")
        assert rc2 == 0 and "[resume] 恢复自 epoch 0, 从 1 续训" in txt2, \
            f"smoke {name}: resume 标记缺失(exit={rc2})"
        terms2, losses2 = _parse_loss_terms(lp2)
        # run_pt 的 epoch 为 0 基编号: 续训 run 只应出现 epoch>=1(即第二个 epoch)
        assert losses2 and all(d["epoch"] >= 1 for d in losses2), \
            f"smoke {name}: 续训 epoch 不连续(出现 epoch<1 = 从头重训)"
        # LP 结构加载检查(missing 仅 fc.*)
        load_encoders(ck / "encoder_group.pth", c["trc"])
        marker.write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
        log(f"smoke {name}: 五项全过(损失有限/开关一致/resume 连续/结构加载/缓存正确)")
        event(f"smoke {name} 通过: 项={sorted(terms)}; config/语料/R峰缓存核验一致")
    # M7 额外端到端 LP 冒烟(cpsc, ~3min)
    e2e = OUTD / "logs" / "smoke_ok_m7_e2e"
    if not e2e.exists():
        ck = smoke_root / "m7"
        feat = OUTD / "feat" / "smoke_m7_cpsc_lp"
        with open_out(LOGD, "smoke_m7_e2e.log", encoding="utf-8") as f:
            subprocess.run(
                [PY, "-u", "run_lp.py", "--data-dir", "data/cpsc", "--num-classes", "9",
                 "--checkpoint", str(ck / "encoder_group.pth"), "--feat-dir", str(feat),
                 "--seed", "0", "--workers", "6", "--trc", "0"],
                stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        assert (feat / "metrics.json").exists(), "smoke m7: 端到端 LP 无 metrics.json"
        e2e.write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
        event("smoke m7 端到端 CPSC LP 通过(结构加载+预测链路)")
    log("smoke 阶段完成")


# ---------------------------------------------------------------- 阶段: ft10fill(P0-1)
def stage_ft10fill():
    nc_of = dict(DOMAINS)
    for name in ("m3", "m4", "m6"):
        for ds in ("cpsc", "chapman"):
            for seed in SEEDS:
                if not run_ft10(name, seed, ds, nc_of[ds]):
                    sys.exit(1)
    event("P0-1 ft10fill 完成: M3/M4/M6 × cpsc/chapman × s024 = 18 格补齐")


# ---------------------------------------------------------------- 阶段: 配置主循环
def run_config(name):
    c = CONFIGS[name]
    nc_of = dict(DOMAINS)
    seeds = c.get("seeds", SEEDS)
    for seed in seeds:
        if not run_pt(name, seed):
            event(f"{name} seed{seed} PT 失败, 链终止(logs/{name}_pt_seed{seed}.log)")
            sys.exit(1)
        for ds in c["lp"]:
            if not run_lp(name, seed, ds, nc_of[ds]):
                sys.exit(1)
        for ds in c["ft"]:
            if not run_ft10(name, seed, ds, nc_of[ds]):
                sys.exit(1)
        if c.get("noise27"):
            noise_eval(name, seed, cells="27")
        if c.get("noise9"):
            noise_eval(name, seed, cells="9")
    event(f"{name} seeds{seeds} 全下游完成(LP{len(c['lp'])}域/FT{len(c['ft'])}域/"
          f"{'27格' if c.get('noise27') else ('9格' if c.get('noise9') else '无噪')})")


def stage_ft10ext():
    """可选(默认不跑): M2/M5 补 cpsc/chapman FT10——protocol_frozen §9 缺口。"""
    nc_of = dict(DOMAINS)
    for name in ("m2", "m5"):
        for ds in ("cpsc", "chapman"):
            for seed in SEEDS:
                if not run_ft10(name, seed, ds, nc_of[ds]):
                    sys.exit(1)
    event("ft10ext(可选扩展)完成: M2/M5 × cpsc/chapman × s024")


# ---------------------------------------------------------------- 阶段: 外部运行器
def _run_sub(script, *extra):
    with open_out(LOGD, Path(script).stem + ".log", encoding="utf-8") as f:
        rc = subprocess.run([PY, "-u", script, *extra], stdout=f,
                            stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    if rc != 0:
        event(f"{script} 失败 exit={rc}(见 logs/{Path(script).stem}.log)")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES),
                    help="逗号分隔, 顺序敏感; 可选 ft10ext")
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        assert s in STAGES + ["ft10ext"], s
    for sub in ("feat", "predictions", "ft", "logs", "multilib"):
        (OUTD / sub).mkdir(parents=True, exist_ok=True)
    import os
    (OUTD / "logs" / "chain.pid").write_text(str(os.getpid()), encoding="utf-8")
    log(f"ABC 链启动: stages={stages} 解释器={PY} pid={os.getpid()}")
    for s in stages:
        log(f"=== 阶段 {s} ===")
        if s == "audit":
            stage_audit()
        elif s == "smoke":
            stage_smoke()
        elif s == "ft10fill":
            stage_ft10fill()
        elif s == "ft10ext":
            stage_ft10ext()
        elif s in CONFIGS:
            run_config(s)
        elif s == "n1n2":
            run_config("n1")
            run_config("n2")
        elif s == "bench":
            _run_sub("runlog/ABC_20261015/run_abc_bench.py")
            event("P0-4 效率基准完成: efficiency.csv")
        elif s == "stats0":
            mk = OUTD / "logs" / "stats0_done"
            if mk.exists():
                log("stats0: 已有完成标记, 跳过")
            else:
                _run_sub("runlog/ABC_20261015/run_abc_stats.py", "--families", "p0")
                mk.write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
        elif s == "stats":
            mk = OUTD / "logs" / "stats_done"
            if mk.exists():
                log("stats: 已有完成标记, 跳过")
            else:
                _run_sub("runlog/ABC_20261015/run_abc_stats.py", "--families", "all")
                mk.write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
                event("全量统计完成: paired/factorial/ledger/三表")
        elif s == "multilib":
            _run_sub("runlog/ABC_20261015/run_abc_multilib.py")
            event("P1-3 多标签敏感性完成: multilib/ 与 multilib_results.csv")
        elif s == "report":
            # 10-17 优化: 报告只从既有 CSV 装配, 不重跑 11h 统计(值班#25 记录)
            _run_sub("runlog/ABC_20261015/run_abc_stats.py", "--report-only")
            event("ABC最终报告.md 生成(report-only 模式)")
    log("ABC 链(本轮调度部分)完成")
    event("ABC 链阶段完成: " + ",".join(stages))


if __name__ == "__main__":
    main()
