# -*- coding: utf-8 -*-
"""JOINT_20261013 N1/N2 反事实开关单测(apply_n1n2.py 落地补丁后由其触发运行)。

N1 = --dv-rand-window 1: 心拍窗口起点均匀随机(窗数=真 R 峰数, RR 目标不动);
N2 = --rr-shuffle 1: RR 目标在有效样本间固定置换(种子123, 掩码/心拍视图不动)。
红线: 两开关默认关时行为与原版逐位一致(签名默认值+调用点守卫+源码守卫)。
"""
import inspect
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # tests/ 在仓库根下一层(勿用 parents[2]——那是 runlog/ 深度约定, 10-14 00:1x 守夜器事故#4 根因)


def _mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dv_dataset_t", ROOT / "data_utils" / "dv_dataset.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_rand_beat_view_shape_determinism():
    m = _mod()
    raw = np.random.RandomState(7).randn(8, 2048).astype(np.float32)
    bv = m.rand_beat_view(raw, 5, np.random.RandomState(42))
    assert bv.shape == (8, m.W_BEAT) and bv.dtype == np.float32
    # 同种子逐位可复现
    assert np.array_equal(bv, m.rand_beat_view(raw, 5, np.random.RandomState(42)))
    # 不同种子不同(均匀随机起点, 固定种子对拍)
    assert not np.array_equal(bv, m.rand_beat_view(raw, 5, np.random.RandomState(43)))
    # 与 R 峰对齐版不同(合成峰集中, 随机窗几乎不可能重合)
    peaks = np.array([500, 700, 900, 1100, 1300])
    assert not np.array_equal(bv, m.beat_view(raw, peaks))


def test_rand_beat_view_short_signal():
    m = _mod()
    raw = np.zeros((8, 100), dtype=np.float32)  # T < w
    assert m.rand_beat_view(raw, 3, np.random.RandomState(0)) is None


def test_build_rr_perm_bijection_determinism():
    m = _mod()
    valid = list(range(10))
    p1 = m.build_rr_perm(valid, seed=123)
    p2 = m.build_rr_perm(valid, seed=123)
    assert p1 == p2
    # 有效集合上的双射, 不映射到集合外
    assert set(p1.keys()) == set(valid) and set(p1.values()) == set(valid)
    # 与另一种子不同(n=10 固定种子对拍)
    assert p1 != m.build_rr_perm(valid, seed=124)
    assert m.build_rr_perm([], seed=123) == {}


def test_flags_default_off_signature():
    m = _mod()
    sig = inspect.signature(m.DVDataset.__init__)
    assert sig.parameters["rand_window"].default is False
    assert sig.parameters["rr_shuffle"].default is False


def test_run_pt_source_guards():
    """源码守卫(W8 E6 先例): 新旗标默认 0, Stage4 调用点透传, 默认路径零行为变化。"""
    src = (ROOT / "run_pt.py").read_text(encoding="utf-8")
    assert src.count("'--dv-rand-window', default=0") == 1
    assert src.count("'--rr-shuffle', default=0") == 1
    assert src.count("rand_window=bool(int(getattr(args, 'dv_rand_window', 0) or 0))") == 1
    assert src.count("rr_shuffle=bool(int(getattr(args, 'rr_shuffle', 0) or 0))") == 1
    dsrc = (ROOT / "data_utils" / "dv_dataset.py").read_text(encoding="utf-8")
    # 默认关时 __getitem__ 的 N1/N2 分支均被短路(rand_window False / rr_src None); 断言串=补丁实际行
    assert "if self.rand_window and bv is not None:" in dsrc
    assert "if self.rr_src is not None and self.rr_src.get(idx) is not None:" in dsrc
