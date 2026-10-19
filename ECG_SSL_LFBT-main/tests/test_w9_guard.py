# -*- coding: utf-8 -*-
"""W9 任务书 §五: metrics_ext 预测目录守卫加白 runlog/W9/ 的门禁单测。

证明三件事(任务书原文):
  1. W9 允许(runlog/W9/predictions/xx 可写);
  2. W2 仍拒绝(冻结只读);
  3. 任意其他路径(仓库外 / runlog 根 / W10 不存在段)仍拒绝。
同时回归: W3..W8 / JOINT_20261013 / ABC_20261015 仍允许(默认行为不变)。
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metrics_ext as mx


def _guard(path):
    """只触发守卫本身, 不真正写盘。"""
    mx._assert_w3_dir(Path(path))


def test_w9_allowed():
    for p in ("runlog/W9/predictions/clocs_cpsc_s0",
              "runlog/W9/predictions/m2_chapman_s4",
              "runlog\\W9\\predictions\\x"):
        _guard(p)  # 不抛异常即通过


def test_w9_real_abs_path_allowed(tmp_path=None):
    # resolve 后的绝对路径也必须放行(运行时实际传入形态)
    here = Path(__file__).resolve()
    p = here.parents[1] / "runlog" / "W9" / "predictions" / "t"
    _guard(p)


def test_w2_still_rejected():
    for p in ("runlog/W2/predictions/x", "runlog\\W2\\x", "F:/x/runlog/W2/p"):
        try:
            _guard(p)
        except ValueError as e:
            assert "W2" in str(e)
        else:
            raise AssertionError(f"W2 未被拒绝: {p}")


def test_other_paths_still_rejected():
    bad = ["F:/some/where/else", "runlog/predictions/x", "runlog/W10/predictions/x",
           "data/pred_out", "C:/Temp/out"]
    with tempfile.TemporaryDirectory() as td:
        bad.append(str(Path(td) / "w9_like" / "predictions"))
        for p in bad:
            try:
                _guard(p)
            except ValueError:
                pass
            else:
                raise AssertionError(f"非法路径未被拒绝: {p}")


def test_legacy_allowlists_unchanged():
    ok = ["runlog/W3/p/x", "runlog/W4/p/x", "runlog/W5/p/x", "runlog/W6/p/x",
          "runlog/W7/p/x", "runlog/W8/p/x",
          "runlog/JOINT_20261013/p/x", "runlog/ABC_20261015/p/x"]
    for p in ok:
        _guard(p)


if __name__ == "__main__":
    test_w9_allowed()
    test_w9_real_abs_path_allowed()
    test_w2_still_rejected()
    test_other_paths_still_rejected()
    test_legacy_allowlists_unchanged()
    print("test_w9_guard: ALL PASS (5/5)")
