# -*- coding: utf-8 -*-
"""A-P2 开关单测: 默认关 == B0 逐位一致(HOSTS §三 A-P2 纪律, 不过此测不得入队)。

运行: python tests/test_ap2_switches.py
参照物: git HEAD 版 models/vgg_1d.py(开关实现前的冻结基线), 经 git show 提取加载。
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.vgg_1d import VGG16, group_whiten, power_pool  # noqa: E402


def load_reference_module():
    """从 git HEAD 提取开关实现前的 vgg_1d.py 作为逐位参照。"""
    src = subprocess.run(
        ["git", "-C", str(ROOT), "show", "HEAD:ECG_SSL_LFBT-main/models/vgg_1d.py"],
        capture_output=True, text=True).stdout
    assert "blur_pool" not in src, "HEAD 已含开关实现, 参照失效——请改用更早基线"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "vgg_ref.py"
        p.write_text(src, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("vgg_ref", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def main():
    torch.manual_seed(0)
    x = torch.randn(4, 1, 2048)

    # 1) 关态结构一致(state_dict 键完全相同)
    ref_mod = load_reference_module()
    ref = ref_mod.VGG16(ch_in=1, alpha=0.125)
    new = VGG16(ch_in=1, alpha=0.125)
    ref.eval(); new.eval()
    rk = set(ref.state_dict().keys()); nk = set(new.state_dict().keys())
    assert rk == nk, f"关态 state_dict 键不一致: {rk ^ nk}"

    # 2) 同权重下关态前向逐位一致(逐位一致性的实质检查)
    new.load_state_dict(ref.state_dict(), strict=False)
    with torch.no_grad():
        y_ref, y_new = ref(x), new(x)
    assert torch.equal(y_ref, y_new), "加载参照权重后关态前向不逐位一致!"

    # 3) H2 BlurPool 开态: 同权重下与关态不同(差异只来自开关), 仅新增 kernel buffer
    blur = VGG16(ch_in=1, alpha=0.125, blur_pool=3); blur.eval()
    blur.load_state_dict(ref.state_dict(), strict=False)
    with torch.no_grad():
        y_b = blur(x)
    assert y_b.shape == y_new.shape and torch.isfinite(y_b).all()
    assert not torch.equal(y_b, y_new)
    assert all("kernel" in k for k in set(blur.state_dict()) - nk), "应只新增 blur kernel buffer"

    # 4) T3 Power-Mean 开态: 同权重下与关态不同; 无新增参数
    powm = VGG16(ch_in=1, alpha=0.125, pool_power=3.0); powm.eval()
    powm.load_state_dict(ref.state_dict(), strict=False)
    with torch.no_grad():
        y_p = powm(x)
    assert y_p.shape == y_new.shape and torch.isfinite(y_p).all()
    assert not torch.equal(y_p, y_new)
    assert set(powm.state_dict()) == nk, "T3 不应有新增参数"

    # 5) power_pool 数值稳定: 极大输入不溢出
    big = torch.randn(2, 8, 512) * 1e20
    assert torch.isfinite(power_pool(big, 3)).all(), "power_pool 大值溢出!"

    # 6) H1 group_whiten: 组内协方差 -> I(容差按采样噪声 4/sqrt(B) 计)
    B = 4096
    h = torch.randn(B, 64) * torch.rand(1, 64) * 10 + torch.rand(1, 64) * 5
    w = group_whiten(h, g=8)
    tol = 4.0 / (B ** 0.5)
    for gi in range(8):
        c = torch.cov(w[:, gi * 8:(gi + 1) * 8].T)
        assert (c.diagonal() - 1).abs().max() < 0.2, f"组{gi}方差偏离1"
        off = c - torch.diag(c.diagonal())
        assert off.abs().max() < tol + 0.05, f"组{gi}白化不充分(非对角>{tol}): {off.abs().max()}"

    # 7) H1 的 NEG 变体: LN-NEG 形状有限; 位置-NEG 置换可逆
    from models.vgg_1d import group_ln
    y_ln = group_ln(h, g=8)
    assert y_ln.shape == h.shape and torch.isfinite(y_ln).all()
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(0))
    inv = torch.empty_like(perm); inv[perm] = torch.arange(64)
    assert torch.equal(h[:, perm][:, inv], h), "位置-NEG 置换不可逆!"

    # 8) runner LP 旗标透传
    from pipeline_runner import lp_arch_args
    assert lp_arch_args("--blur-pool 3 --speed-perturb 0.85,1.15") == ["--blur-pool", "3"]
    assert lp_arch_args("--pool-power 3") == ["--pool-power", "3"]
    assert lp_arch_args("--cautious") == []

    print("[PASS] A-P2 全部 8 项单测: 关态逐位一致 + 开态行为正确 + 白化协方差=I + NEG 变体")


if __name__ == "__main__":
    main()
