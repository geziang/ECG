# -*- coding: utf-8 -*-
"""apply_n1n2.py — N1/N2 反事实开关的热文件安全落地器(detached 守夜)。

纪律背景: Windows DataLoader 每 epoch 从磁盘重 import —— run_pt.py 在跑时
严禁改被 import 的 .py(整机重启恢复手册红线)。本脚本等到"无 run_pt.py 进程"
的窗口(链在 LP/FT10 段或段间)才写入, 每次写入前二次确认, 锚点 count==1
(assert 纪律), 失败即回滚原件。

流程: 锚点干跑校验 → 等安全窗口 → 写 dv_dataset.py + run_pt.py →
py_compile → pytest tests/test_joint_n1n2.py → commit+push(尽力) →
touch .n1n2_ready(值班 cron 见此标记且链 phase-1 完成后重启链接跑 n1/n2)。
失败: 还原 .orig → events 记 FAIL → exit 3(链不受影响)。
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
OUTD = ROOT / "runlog/JOINT_20261013"
DV = ROOT / "data_utils" / "dv_dataset.py"
RP = ROOT / "run_pt.py"
READY = OUTD / ".n1n2_ready"
MAX_WAIT_S = 14 * 3600


def log(msg):
    line = f"[n1n2 {time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUTD / "logs" / "apply_n1n2.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def event(msg):
    with open(OUTD / "events.md", "a", encoding="utf-8") as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} {msg}\n")


def _cim_run_pt_count():
    """CIM 数 run_pt python(会随机漏行——09-20 已知机器病, 只作兜底通道)。"""
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' "
         "-and $_.CommandLine -like '*run_pt.py*' }).Count"],
        capture_output=True, text=True, timeout=90)
    try:
        return int(r.stdout.strip() or "0")
    except ValueError:
        return 99  # 查询异常按在跑处理(保守)


def pt_alive():
    """双通道判活: A=链分段日志(链进程自己写, 免疫 WMI 抽风); B=CIM 兜底。

    A: chain.out.log 最后一次 'PT 启动' 行之后若无 'PT: exit=' 行 => PT 在跑。
    任一通道报告在跑即视为在跑(漏检方向=安全方向)。
    """
    try:
        tail = (OUTD / "logs" / "chain.out.log").read_text(encoding="utf-8",
                                                           errors="replace").splitlines()
        starts = [i for i, l in enumerate(tail) if " PT 启动:" in l]
        exits = [i for i, l in enumerate(tail) if "PT: exit=" in l]
        if starts and (not exits or starts[-1] > exits[-1]):
            return True
    except OSError:
        return True  # 日志读不到, 保守
    return _cim_run_pt_count() > 0


# ---------------- 补丁定义(锚点均已干跑核验 count==1) ----------------
DV_INSERT_BEFORE_CLASS = ("class DVDataset:", """
def rand_beat_view(raw, n_windows, rng, w=W_BEAT):
    \"\"\"N1 反事实(JOINT N1): 与 beat_view 同窗长同窗数, 起点均匀随机, 无 R 峰对齐。

    rng: np.random.RandomState; 逐记录确定性由调用方用 crc32(stem) 播种。
    \"\"\"
    T = raw.shape[1]
    if n_windows < 3 or T < w:
        return None
    starts = rng.randint(0, T - w + 1, size=int(n_windows))
    segs = [raw[:, s:s + w] for s in starts]
    return np.mean(segs, axis=0).astype(np.float32)


def build_rr_perm(valid_idx, seed=123):
    \"\"\"N2 反事实(JOINT N2): valid_idx 内固定置换, 返回 idx->源 idx 映射。

    第 k 个有效样本拿第 perm[k] 个有效样本的 RR 统计; 自映射不排除
    (n≈3.5 万下占比可忽略, 属目标噪声一部分)。
    \"\"\"
    if not valid_idx:
        return {}
    perm = np.random.RandomState(seed).permutation(len(valid_idx))
    return {valid_idx[k]: valid_idx[int(perm[k])] for k in range(len(valid_idx))}


""")

DV_INIT_SIG = (
    "def __init__(self, root, transform1, transform2, rpeak_npz):",
    "def __init__(self, root, transform1, transform2, rpeak_npz,\n"
    "                 rand_window=False, rr_shuffle=False):")

DV_NVALID = ("        self.n_valid = len(stats)", """        self.n_valid = len(stats)
        self.rand_window = bool(rand_window)  # N1: 默认关=原行为
        if rr_shuffle:  # N2: idx->源idx 固定置换(顺序=base.samples, 种子123)
            from pathlib import Path as _P
            stats_by_idx = [rr_stats(self.rp.get(_P(self.base.samples[i][0]).stem))
                            for i in range(len(self.base))]
            self.rr_src = build_rr_perm(
                [i for i, s in enumerate(stats_by_idx) if s is not None])
        else:
            self.rr_src = None""")

DV_GETITEM = ("""        bv = beat_view(raw, peaks)
        rs = rr_stats(peaks)""", """        bv = beat_view(raw, peaks)
        rs = rr_stats(peaks)
        if self.rand_window and bv is not None:
            # N1: 窗数=真 R 峰数, 起点均匀随机, 逐记录确定性种子(crc32(stem))
            import zlib
            rng = np.random.RandomState(zlib.crc32(key.encode()) % (2 ** 31))
            bv = rand_beat_view(raw, len(peaks), rng)""")

DV_ELSE = ("""            beat = torch.from_numpy(bv)
            rr3 = torch.tensor((np.asarray(rs) - self.mu) / self.sd, dtype=torch.float32)""", """            beat = torch.from_numpy(bv)
            if self.rr_src is not None and self.rr_src.get(idx) is not None:
                # N2: 目标换成置换源记录的 RR 统计(掩码与心拍视图不动)
                rs_src = rr_stats(self.rp.get(
                    Path(self.base.samples[self.rr_src[idx]][0]).stem))
                if rs_src is not None:
                    rs = rs_src
            rr3 = torch.tensor((np.asarray(rs) - self.mu) / self.sd, dtype=torch.float32)""")

DV_PATCHES = [DV_INSERT_BEFORE_CLASS, DV_INIT_SIG, DV_NVALID, DV_GETITEM, DV_ELSE]

RP_ARGS = ("""parser.add_argument('--rpeak-npz', default='data/pt_rpeaks.npz', type=str,
                    help='R 峰缓存文件(gqrs 预计算, 见 runlog/prep_rpeaks.py)')""", """parser.add_argument('--rpeak-npz', default='data/pt_rpeaks.npz', type=str,
                    help='R 峰缓存文件(gqrs 预计算, 见 runlog/prep_rpeaks.py)')
parser.add_argument('--dv-rand-window', default=0, type=int,
                    help='JOINT N1 反事实: 心拍窗起点均匀随机(RR 仍用真 R 峰); 默认 0=原行为')
parser.add_argument('--rr-shuffle', default=0, type=int,
                    help='JOINT N2 反事实: RR 目标有效样本间固定置换(种子123); 默认 0=原行为')""")

RP_CALL = ("        dataset = DVDataset(args.data_dir, t, t2, rpeak_npz)", """        dataset = DVDataset(args.data_dir, t, t2, rpeak_npz,
                            rand_window=bool(int(getattr(args, 'dv_rand_window', 0) or 0)),
                            rr_shuffle=bool(int(getattr(args, 'rr_shuffle', 0) or 0)))""")

RP_PATCHES = [RP_ARGS, RP_CALL]


def apply_patches(path, patches, orig_backup):
    src = path.read_text(encoding="utf-8")
    for old, new in patches:
        assert src.count(old) == 1, f"{path.name}: 锚点非唯一: {old[:50]!r}"
    backup = src
    patched = src
    for old, new in patches:
        patched = patched.replace(old, new, 1)
    # 写盘前先编译校验补丁后内容(语法错误绝不落盘)
    compile(patched, f"{path.name}(patched)", "exec")
    if pt_alive():  # 写入前最后一刻复检
        raise RuntimeError("安全窗口丢失(PT 复活), 放弃本轮写入")
    path.write_text(patched, encoding="utf-8")
    orig_backup.append((path, backup))


def main():
    if READY.exists():
        log("已完成过, 幂等退出")
        return 0
    # 1. 干跑锚点(不写文件)
    dv_src = DV.read_text(encoding="utf-8")
    rp_src = RP.read_text(encoding="utf-8")
    for old, _ in DV_PATCHES:
        assert dv_src.count(old) == 1, f"dv 锚点漂移: {old[:50]!r}"
    for old, _ in RP_PATCHES:
        assert rp_src.count(old) == 1, f"rp 锚点漂移: {old[:50]!r}"
    log(f"锚点干跑 7/7 唯一, 开始守候安全窗口(无 run_pt.py 进程, 最长 {MAX_WAIT_S//3600}h)")
    # 2. 等安全窗口
    t0 = time.time()
    while True:
        while pt_alive():
            if time.time() - t0 > MAX_WAIT_S:
                log("超时未等到安全窗口, 放弃(链不受影响)")
                event("N1/N2 落地器: 等安全窗口超时放弃, 待人工处置")
                return 2
            time.sleep(60)
        time.sleep(20)  # 双重确认: 防瞬时探针抽风, 再探一次仍无 PT 才动笔
        if not pt_alive():
            break
    log("进入安全窗口(双通道双确认无 run_pt 进程), 开始写入")
    backups = []
    try:
        apply_patches(DV, DV_PATCHES, backups)
        apply_patches(RP, RP_PATCHES, backups)
        subprocess.run([PY, "-m", "py_compile", str(DV), str(RP)], check=True,
                       capture_output=True, timeout=120)
        log("py_compile 通过, 跑单测")
        tr = subprocess.run([PY, "-m", "pytest", "-q",
                             "tests/test_joint_n1n2.py"],
                            cwd=str(ROOT), capture_output=True, text=True, timeout=600)
        log(f"pytest rc={tr.returncode}: {tr.stdout.strip().splitlines()[-1] if tr.stdout.strip() else '(无输出)'}")
        if tr.returncode != 0:
            raise RuntimeError(f"单测未过:\n{tr.stdout[-2000:]}\n{tr.stderr[-1000:]}")
    except Exception as e:
        for path, backup in backups:
            path.write_text(backup, encoding="utf-8")
        subprocess.run([PY, "-m", "py_compile", str(DV), str(RP)], capture_output=True)
        log(f"FAIL 已回滚原件: {type(e).__name__}: {e}")
        event(f"N1/N2 落地器 FAIL 已回滚: {type(e).__name__}(详见 logs/apply_n1n2.log), 链不受影响")
        return 3
    # 4. 提交 + 就绪标记
    subprocess.run(["git", "-C", str(ROOT), "add", str(DV), str(RP),
                    "tests/test_joint_n1n2.py", "runlog/JOINT_20261013"],
                   capture_output=True)
    subprocess.run(["git", "-C", str(ROOT), "commit", "-q", "-m",
                    "JOINT N1/N2 落地(主机A): +--dv-rand-window/--rr-shuffle 反事实开关(默认关==原行为, 单测5/5+源码守卫) — 安全窗口写入, 锚点count==1, apply_n1n2.py 自动落地"],
                   capture_output=True)
    subprocess.run(["git", "-C", str(ROOT), "-c", "http.lowSpeedLimit=0",
                    "-c", "http.lowSpeedTime=999", "push", "-q", "origin", "main"],
                   capture_output=True, timeout=180)
    READY.write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
    log("落地完成: 补丁+单测+提交, .n1n2_ready 已标记")
    event("N1/N2 开关落地完成(单测过, 已提交): 链 phase-1(m3..m6)完成后由值班重启接跑 n1/n2(先 s0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
