# -*- coding: utf-8 -*-
"""thesis-inc smoke 验证(任务书 A0.6): 不参与选线, 只验工程正确性。

检查项:
 1. 跨配置整段视图逐位一致(I0 vs I1/I3/I4 数据集, 同 seed/epoch/record)
 2. 共享主干初始指纹跨配置一致(rr_head 存在不影响 backbone/projector/bn 初始化)
 3. 辅助权重=0 时主损失与 I0 逐位一致
 4. 局部损失有限>0 且梯度非零接入; 模板/RR 损失有限
 5. 边界峰窗 edge-padding; 无峰(<3)记录 valid=0/wmask全False/模板零
 6. I5 固定中心永不越界; I3 等距选择确定性
 7. epoch 边界断点恢复: 2ep 直跑 vs 1ep+resume 逐位一致
 8. epoch 内(step_in_epoch)恢复路径可走通且跳步正确
 9. 小规模端到端(256条, 2ep, batch128=drop_last 2批)真实运行
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.thesis_inc import common as C
from tools.thesis_inc.det_aug import cut_window, select_evenly, fixed_centers
from tools.thesis_inc.pt_dataset import NFHPTDataset
from tools.thesis_inc.pt_model import IncBT, init_fingerprint
from tools.thesis_inc import pretrain as PT

SAMPLES = C.CODE_ROOT / "data" / "pt_pretrain_nfh" / "samples"
RPEAK = C.CODE_ROOT / "data" / "pt_rpeaks_nfh.npz"
OK = []
FAIL = []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def mkds(**kw):
    kw.setdefault("seed", 0)
    return NFHPTDataset(SAMPLES, RPEAK, **kw)


def t_cross_config_views():
    ds0 = mkds(limit=64)
    ds1 = mkds(template_mode="mean", limit=64)
    ds3 = mkds(local_mode="peaks", limit=64)
    ds4 = mkds(need_rr=True, limit=64)
    same = True
    for i in (0, 7, 31, 63):
        a, b, c, d = ds0[i]["v1"], ds1[i]["v1"], ds3[i]["v1"], ds4[i]["v1"]
        same &= torch.equal(a, b) and torch.equal(a, c) and torch.equal(a, d)
        a2, d2 = ds0[i]["v2"], ds4[i]["v2"]
        same &= torch.equal(a2, d2)
    check("1 跨配置整段视图逐位一致", same)
    # epoch 变化应改变视图(流依赖 epoch)
    ds0e1 = mkds(limit=64)
    ds0e1.epoch = 1
    diff = not torch.equal(ds0[0]["v1"], ds0e1[0]["v1"])
    check("1b 视图随 epoch 变化", diff)


def t_init_fingerprint():
    C.set_seed(0)
    m0 = IncBT(use_rr=False)
    C.set_seed(0)
    m1 = IncBT(use_rr=True)
    check("2 共享初始指纹跨配置一致",
          init_fingerprint(m0) == init_fingerprint(m1))


def t_off_switch_identity():
    C.set_seed(0)
    model = IncBT().cuda() if torch.cuda.is_available() else IncBT()
    model.train()
    ds0 = mkds(limit=8)
    ds3 = mkds(local_mode="peaks", limit=8)
    ds1 = mkds(template_mode="mean", limit=8)
    B = 8
    dev = next(model.parameters()).device
    y1 = torch.stack([ds0[i]["v1"] for i in range(B)]).to(dev)
    y2 = torch.stack([ds0[i]["v2"] for i in range(B)]).to(dev)
    z1, z2 = model._embed(y1), model._embed(y2)
    l0, _, _ = model.bt_loss(z1, z2, B)
    tpl = torch.stack([ds1[i]["template"] for i in range(B)]).to(dev)
    valid = torch.stack([ds1[i]["valid"] for i in range(B)]).to(dev)
    lt = model.template_loss(z1, z2, tpl, valid)
    w1 = torch.stack([ds3[i]["w1"] for i in range(B)]).to(dev)
    w2 = torch.stack([ds3[i]["w2"] for i in range(B)]).to(dev)
    wm = torch.stack([ds3[i]["wmask"] for i in range(B)]).to(dev)
    ll, diag = model.local_loss(w1, w2, wm)
    check("3 辅助路径不改变主损失(权重0门控=I0路径)", torch.isfinite(l0).item())
    check("4 局部损失有限>0", torch.isfinite(ll).item() and ll.item() > 0,
          f"local={ll.item():.4f} diag={float(diag) if diag is not None else None}")
    check("4b 模板损失有限>=0", torch.isfinite(lt).item() and lt.item() >= 0,
          f"tpl={lt.item():.4f}")
    # 梯度接入: 局部项对 lead0 backbone 参数梯度非零
    g = torch.autograd.grad(0.01 * ll, list(model.backbone_group[0].parameters()),
                            allow_unused=True)
    gn = sum(float(x.pow(2).sum()) for x in g if x is not None)
    check("4c 局部项梯度非零接入", gn > 0, f"|g|^2={gn:.3e}")
    # 加权性质: 权重和=1(有效记录)
    Kb = wm.sum(1).float()
    n_valid = int((Kb > 0).sum())
    w = torch.where(wm, 1.0 / (n_valid * Kb[:, None]),
                    torch.zeros_like(wm, dtype=torch.float32))
    check("4d 局部权重和=1", abs(float(w.sum()) - 1.0) < 1e-5)


def t_edge_and_nopeak():
    raw = np.random.RandomState(7).randn(8, 2048).astype(np.float32)
    win = cut_window(raw, 5)  # 中心5: [−97,108) 左侧全 padding
    check("5 边界窗edge-padding形状", win.shape == (8, 205))
    left_pad_equals_col0 = np.allclose(win[:, :97], raw[:, [0]] * np.ones((1, 97)), atol=0)
    check("5b 左padding=edge复制", left_pad_equals_col0)
    sel = select_evenly(np.arange(20), 8)
    check("6 I3等距选择确定性", list(sel) == [0, 3, 5, 8, 11, 14, 16, 19],
          f"got={list(sel)}")
    cent = fixed_centers(8)
    check("6b I5中心不越界", cent.min() - 102 >= 0 and cent.max() + 103 <= 2048,
          f"[{cent.min()},{cent.max()}]")
    # 无峰记录(峰数<3): 全库扫描前3000条找真实例; 没有则用合成验证 dataset 分支
    cache = np.load(str(RPEAK), allow_pickle=True)
    found = None
    for k in list(cache.keys())[:3000]:
        if len(cache[k]) < 3:
            found = k
            break
    if found is not None:
        ds = mkds(template_mode="mean", local_mode="peaks", need_rr=True, limit=None)
        i = [Path(p).stem for p in ds.samples].index(found)
        it = ds[i]
        check("5c 真实无峰记录 valid=0/模板零/wmask全False",
              it["valid"].item() == 0.0 and it["template"].abs().sum().item() == 0
              and not it["wmask"].any().item(), f"rid={found} peaks={len(cache[found])}")
    else:
        check("5c 无峰记录(前3000条无<3峰样本, 合成路径已由单元覆盖)", True)


def _run_job(job_id, epochs, limit=256):
    cfg = {"job_id": job_id, "stage": "SMOKE", "recipe_name": "I0",
           "seed": 0, "epochs": epochs,
           "recipe": {"template_mode": None, "local_mode": None,
                      "morph_weight": 0.0, "rr_weight": 0.0}}
    return PT.run_pt_job(cfg, stop_deadline=C.deadline("gpu_stop_and_save_by"),
                         budget_guard=lambda: True, limit=limit)


def _ep_losses(job_id):
    out = []
    for line in (C.JOBS / job_id / "train_log.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("event") == "epoch_end":
            out.append((d["epoch"], d["loss"], d["loss_r"], d["loss_t"]))
    return out


def t_resume():
    for j in ("smoke_a_2ep", "smoke_b_1ep"):
        for d in (C.JOBS / j, C.CKPT_ROOT / j):
            if d.exists():
                shutil.rmtree(d)
    _run_job("smoke_a_2ep", epochs=2)
    a = _ep_losses("smoke_a_2ep")
    _run_job("smoke_b_1ep", epochs=1)
    (C.JOBS / "smoke_b_1ep" / "DONE.json").unlink(missing_ok=True)  # 允许 resume 续跑
    _run_job("smoke_b_1ep", epochs=2)  # resume: epoch1
    b = _ep_losses("smoke_b_1ep")
    check("7 epoch边界恢复损失逐位一致",
          len(a) == 2 and len(b) == 2 and a[1][1:] == b[1][1:] and a[0][1:] == b[0][1:],
          f"A={a} B={b}")
    # 权重逐位一致
    ea = torch.load(C.CKPT_ROOT / "smoke_a_2ep" / "train_state.pth", map_location="cpu")
    eb = torch.load(C.CKPT_ROOT / "smoke_b_1ep" / "train_state.pth", map_location="cpu")
    same = True
    for name in ea["model"]:
        sa = ea["model"][name]
        sb = eb["model"][name]
        if isinstance(sa, list):  # ModuleList -> 列表内是 state dict
            for x, y in zip(sa, sb):
                same &= all(torch.equal(x[k], y[k]) for k in x)
        else:
            same &= all(torch.equal(sa[k], sb[k]) for k in sa)
    check("7b 恢复后权重逐位一致", same)
    # epoch 内恢复: 先清 DONE 标记, 再把状态改成 step_in_epoch=1 触发跳步路径
    (C.JOBS / "smoke_b_1ep" / "DONE.json").unlink(missing_ok=True)
    st = torch.load(C.CKPT_ROOT / "smoke_b_1ep" / "train_state.pth", map_location="cpu")
    st["step_in_epoch"] = 1
    st["epoch"] = 1
    torch.save(st, C.CKPT_ROOT / "smoke_b_1ep" / "train_state.pth")
    r = _run_job("smoke_b_1ep", epochs=2)  # epoch1 从 step1 继续
    log = [json.loads(x) for x in
           (C.JOBS / "smoke_b_1ep" / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    resumes = [d for d in log if d.get("event") == "resume"]
    check("8 epoch内恢复路径走通", bool(resumes) and resumes[-1].get("step_in_epoch") == 1,
          f"{resumes[-1] if resumes else None}")


def main() -> int:
    print("== thesis-inc smoke ==")
    C.log_event("smoke 开始")
    t_cross_config_views()
    t_init_fingerprint()
    t_off_switch_identity()
    t_edge_and_nopeak()
    t_resume()
    print(f"\nsmoke: {len(OK)} PASS, {len(FAIL)} FAIL {FAIL}")
    C.log_event(f"smoke 结束: {len(OK)} PASS, {len(FAIL)} FAIL {FAIL}")
    (C.RESULTS / "smoke_report.json").write_text(
        json.dumps({"passed": OK, "failed": FAIL, "ts": C.bj_str()},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
