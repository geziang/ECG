# -*- coding: utf-8 -*-
"""run_abc_bench.py — ABC_20261015 P0-4 效率与成本基准(protocol_frozen §7)。

M0–M9 十模型统一: 参数量 / 单记录 MACs(hook 计数 Conv1d+Linear) / PT 墙钟与峰值显存
(manifest 优先, 缺则扫各批次 events.md 的 "PT 完成" 行, 再缺如实 n/a) /
FT10 墙钟 / b=1 推理延迟(预热50 + 5轮×200次, cuda 同步, 中位数+IQR) /
b=64 吞吐 / 推理峰值显存。结构类指标(MACs/参数/延迟)以 seed0 代表, 逐模型记录。
输出 runlog/ABC_20261015/efficiency.csv(整文件重写, 幂等)。
"""
import csv
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "runlog/ABC_20261015"
OUT = OUTD / "efficiency.csv"
IN_LEN = 2048
N_CLASSES = 9  # 推理基准统一用 CPSC 头结构(与 FT 一致的 MultiBranchNet)

MODELS = {  # name -> (seed0 ckpt 目录, trc)
    "m0": ("checkpoint/confirm/c1_seed0", 0),
    "m1": ("checkpoint/confirm/c2_seed0", 1),
    "m2": ("checkpoint/w6_dv/seed0", 0),
    "m3": ("checkpoint/joint_m3/seed0", 1),
    "m4": ("checkpoint/joint_m4_noise/seed0", 1),
    "m5": ("checkpoint/w8_noiseiso/seed0", 0),
    "m6": ("checkpoint/joint_m6/seed0", 1),
    "m7": ("checkpoint/abc_supp/m7_bc/seed0", 0),
    "m8": ("checkpoint/abc_supp/m8_a_beat_c/seed0", 1),
    "m9": ("checkpoint/abc_supp/m9_a_rr_c/seed0", 1),
}
SEEDS = [0, 2, 4]
# PT 墙钟回溯: (model, seed) -> manifest 键 或 events.md 扫描标签
PT_LOOKUP = {
    "m3": "joint_m3", "m4": "joint_m4_noise", "m6": "joint_m6",
    "m7": "m7", "m8": "m8", "m9": "m9",
}
PT_EVENT_TAG = {"m0": "c1", "m1": "c2", "m2": "dv", "m5": "noiseiso"}


def log(msg):
    print(f"[bench {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def build_net(ckpt, trc):
    from models.mbn import MultiBranchNet
    return MultiBranchNet(N_CLASSES, checkpoint=ckpt, trc=trc).cuda().eval()


def count_params(net):
    total = sum(p.numel() for p in net.parameters())
    head = sum(p.numel() for n, p in net.parameters() if n.startswith("fc"))
    return total, total - head, head


def count_macs(net):
    """hook 计数 Conv1d+Linear MACs(单样本); BN/激活不计(口径注明)。"""
    macs = [0]
    hooks = []

    def h_conv(m, i, o):
        oc = o.shape[1] * o.shape[-1]
        macs[0] += oc * (m.in_channels // m.groups) * m.kernel_size[0]

    def h_lin(m, i, o):
        macs[0] += m.in_features * m.out_features

    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(h_conv))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(h_lin))
    with torch.no_grad():
        net(torch.randn(1, 8, IN_LEN).cuda())
    for hk in hooks:
        hk.remove()
    return macs[0]


def latency(net, reps=200, rounds=5):
    x1 = torch.randn(1, 8, IN_LEN).cuda()
    with torch.no_grad():
        for _ in range(50):
            net(x1)
        torch.cuda.synchronize()
        ms_all = []
        for _ in range(rounds):
            t = []
            for _ in range(reps):
                t0 = time.perf_counter()
                net(x1)
                torch.cuda.synchronize()
                t.append((time.perf_counter() - t0) * 1e3)
            ms_all += t
    a = np.array(ms_all)
    return float(np.median(a)), float(np.percentile(a, 75) - np.percentile(a, 25))


def throughput(net, bs=64, reps=30):
    xb = torch.randn(bs, 8, IN_LEN).cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(5):
            net(xb)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            net(xb)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    return bs * reps / dt, peak


def pt_walls(model):
    """返回 {seed: (wall_s, vram_mib, source)}; manifest -> events 扫描 -> 缺。"""
    out = {}
    man_paths = [OUTD / "checkpoints_manifest.json",
                 ROOT / "runlog/JOINT_20261013/checkpoints_manifest.json"]
    key = PT_LOOKUP.get(model)
    if key:
        for mp in man_paths:
            if not mp.exists():
                continue
            m = json.loads(mp.read_text(encoding="utf-8")).get(key, {})
            for s, v in m.items():
                if v.get("pt_wall_s") is not None:
                    out[int(s)] = (v["pt_wall_s"], v.get("vram_peak_mib"),
                                   mp.parent.name)
            if out:
                break
    if len(out) < len(SEEDS) and model in PT_EVENT_TAG:
        tag = PT_EVENT_TAG[model]
        pat = re.compile(rf".*\b{re.escape(tag)}\b.*PT.*?(\d+)min.*?(\d+)MiB")
        for ev in sorted(ROOT.glob("runlog/*/events.md")):
            try:
                txt = ev.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for line in txt.splitlines():
                mm = pat.match(line)
                if not mm:
                    continue
                seed = None
                ms = re.search(r"seed\s*(\d+)|s(\d+)", line)
                if ms:
                    seed = int(ms.group(1) or ms.group(2))
                if seed in SEEDS and seed not in out:
                    out[seed] = (float(mm.group(1)) * 60, int(mm.group(2)),
                                 ev.parent.name)
    return out


def ft10_walls(model):
    man = OUTD / "checkpoints_manifest.json"
    if not man.exists():
        return {}
    m = json.loads(man.read_text(encoding="utf-8")).get(model, {})
    return {int(s): v.get("ft10_cpsc_wall_s")
            for s, v in m.items() if v.get("ft10_cpsc_wall_s") is not None}


def main():
    import torch.backends.cudnn as cudnn
    dev = torch.cuda.get_device_name(0)
    env = (f"{dev};cuda{torch.version.cuda};torch{torch.__version__};"
           f"len{IN_LEN};cudnn{cudnn.version()}")
    log(f"环境: {env}")
    rows = []
    for name, (ckrel, trc) in MODELS.items():
        ck = ROOT / ckrel / "encoder_group.pth"
        if not ck.exists():
            log(f"{name}: checkpoint 缺({ck}), 跳过")
            continue
        net = build_net(ck, trc)
        total, enc, head = count_params(net)
        macs = count_macs(net)
        lat_med, lat_iqr = latency(net)
        thr, vram_inf = throughput(net)
        pw, fw = pt_walls(name), ft10_walls(name)
        pt_wall = (f"{np.mean([v[0] for v in pw.values()])/60:.1f}min"
                   if pw else "n/a")
        pt_ep = (f"{np.mean([v[0] for v in pw.values()])/60/100:.2f}min"
                 if pw else "n/a")
        pt_vr = (f"{int(np.mean([v[1] for v in pw.values() if v[1]]))}MiB"
                 if pw and any(v[1] for v in pw.values()) else "n/a")
        pt_src = ";".join(sorted({v[2] for v in pw.values()})) or "n/a(历史批次未记录)"
        ft10 = (f"{np.mean(list(fw.values()))/60:.1f}min" if fw else "n/a")
        rows.append(dict(model=name, params_total=total, params_encoder=enc,
                         params_head=head, macs_per_record=macs,
                         macs_method="hook Conv1d+Linear(BN/激活不计)",
                         pt_wall_100ep=pt_wall, pt_epoch_mean=pt_ep,
                         pt_vram_peak=pt_vr, pt_source=pt_src,
                         ft10_wall_cpsc=ft10,
                         infer_latency_b1_ms_med=round(lat_med, 3),
                         infer_latency_b1_ms_iqr=round(lat_iqr, 3),
                         infer_throughput_b64_recs_s=round(thr, 1),
                         infer_vram_peak_mib=round(vram_inf, 1), env=env,
                         seeds_pt=len(pw)))
        log(f"{name}: params={total} MACs={macs/1e6:.1f}M lat={lat_med:.2f}ms "
            f"thr={thr:.0f}r/s PT={pt_wall}({pt_src})")
        del net
        torch.cuda.empty_cache()
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"efficiency.csv 写出({len(rows)} 行)")


if __name__ == "__main__":
    main()
