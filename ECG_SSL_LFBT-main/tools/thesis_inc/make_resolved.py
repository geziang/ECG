# -*- coding: utf-8 -*-
"""冻结 protocol_resolved.json: 首次正式训练前写出, 之后重跑必须逐位一致。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.thesis_inc import common as C
from tools.thesis_inc.det_aug import AUG_IMPL_VERSION

R1_RECIPES = {
    "I0": {"template_mode": None, "local_mode": None, "morph_weight": 0.0, "rr_weight": 0.0},
    "I1": {"template_mode": "mean", "local_mode": None, "morph_weight": 0.01, "rr_weight": 0.0},
    "I2": {"template_mode": "median", "local_mode": None, "morph_weight": 0.01, "rr_weight": 0.0},
    "I3": {"template_mode": None, "local_mode": "peaks", "morph_weight": 0.01, "rr_weight": 0.0},
    "I4": {"template_mode": None, "local_mode": None, "morph_weight": 0.0, "rr_weight": 0.1},
    "I5": {"template_mode": None, "local_mode": "fixed", "morph_weight": 0.01, "rr_weight": 0.0},
}


def build() -> dict:
    p = C.PROTOCOL
    dm = json.loads((C.RESULTS / "data_manifest.json").read_text(encoding="utf-8"))
    return {
        "protocol_id": p["protocol_id"],
        "frozen_at": C.bj_str(),
        "git_head": C.git_head(),
        "git_dirty": C.git_status_porcelain(),
        "impl_versions": {"aug": AUG_IMPL_VERSION, "dataset": "nfh-pt-ds-v1",
                          "model": "incbt-v1", "pretrain": "pt-v1", "lp": "lp-v1",
                          "runner": "runner-v1"},
        "paths": {
            "results_dir": str(C.RESULTS), "ckpt_root": str(C.CKPT_ROOT),
            "backup_root": str(C.BACKUP_ROOT),
            "nfh_samples": str(C.CODE_ROOT / "data" / "pt_pretrain_nfh" / "samples"),
            "rpeak_npz": str(C.CODE_ROOT / "data" / "pt_rpeaks_nfh.npz"),
            "lp_train_npz": str(C.ADAPT / "train_data.npz"),
            "lp_val_npz": str(C.ADAPT / "val_data.npz"),
        },
        "pretrain_resolved": {
            "architecture": "per-lead VGG16(alpha=0.125, trc=1 GRN1D, fc=Identity) x8; "
                             "projector 64-2048-2048-2048 (Linear no-bias + BN + ReLU, last Linear no-bias); "
                             "output BN BatchNorm1d(2048, affine=False) per lead",
            "loss": "LFBT Barlow Twins: per lead-pair recompute BN, C=c1.T@c2/batch_size, "
                    "on=(diag-1)^2, off=lambd*offdiag^2, loss=gamma*loss_r+(1-gamma)*loss_t",
            "gamma": 0.8, "lambd": 0.0051,
            "optimizer": "Adam(lr=1e-3, betas=(0.9,0.999), eps=1e-8, wd=0); "
                         "param order = weights(ndim>1) then biases(ndim==1) across "
                         "backbone,projector,bn,rr_head (run_pt 同口径)",
            "batch_size": 128, "drop_last": True, "workers": 4, "precision": "float32",
            "scheduler": None, "pilot_epochs": 20, "full_epochs": 100,
            "seeds": {"pilot": [0], "confirm": [2, 4], "full": [0, 2, 4]},
            "init": "from scratch per seed; set_seed(seed) before construction; "
                    "construction order backbone->projector->bn->rr_head 保证共享部分初始指纹跨配方一致",
            "sample_order": "per-epoch permutation via torch.Generator seeded by "
                            "blake2b(f'perm|{seed}|{epoch}'); 同 seed 跨配方逐位一致; drop_last=True",
            "global_views": {
                "pipeline": "NSTDB NoiseInjector -> RRC-TO (源码 Compose 顺序); ToTensor",
                "noise": {"prob": 0.5, "types": ["bw", "ma", "em"],
                          "snr_db_uniform": [5.0, 20.0], "offset_range": "first_half",
                          "resample": "wfdb 360Hz -> 204.8Hz resample_poly(2048,3600)",
                          "snr_match": "per-lead power ratio, x + n*sqrt(px/(pn*10^(snr/10)))"},
                "rrc_to_params": [0.5, 1.0, 0.0, 0.5],
                "rng": "per-(seed,epoch,record,view g1/g2) blake2b 流(det-aug-v1); "
                       "分布与抽样顺序与源代码一致, 机制从 worker 全局 RNG 改为逐记录确定流; "
                       "保证跨配方一致与断点续训复现",
            },
            "aux_resolved": {
                "valid_min_peaks": 3, "window": {"w": 205, "lo": -102, "hi_exclusive": 103,
                                                 "padding": "edge", "per_beat_norm": False},
                "I1_I2_template": "全部有效缓存峰取窗后逐点 mean(I1)/median(I2); "
                                  "一致性=复刻 _dv_consistency(逐对重算BN, on-diag, 两整段视图平均)",
                "I3_peaks_selection": "K=min(峰数,8); 超过8按有序峰索引 round(linspace(0,P-1,8))",
                "I5_fixed_centers": "floor((j+0.5)*2048/K)",
                "local_views": "同一组原始窗两份, 各自仅 NoiseInjector(prob0.5), 无时间裁剪/平移/翻转; "
                               "RNG=(seed,epoch,record,loc{j}v{1,2}) 独立于整段流",
                "local_objective": "共享 backbone+projector; 权重 w_bj=1/(N_valid*K_b); "
                                   "加权标准化 sqrt(var+1e-4); C=Z1s.T diag(w) Z2s; "
                                   "逐导联 on=(diag-1)^2 + 0.0051*off^2; 八导联平均; "
                                   "N_valid<2 跳过该批局部项并计数; padding 窗(峰近边界)按 edge 填充后照常入行",
                "rr": "rr_head=Linear(64,32)+ReLU+Linear(32,3), 输入视图1首导(II) 64维GAP特征; "
                      "目标 [mean_RR,SDNN,RMSSD] 秒, 全库 z-score (mu/sd 实测见 data_manifest)",
                "rr_mu": dm["nfh"]["rr_mu"], "rr_sd": dm["nfh"]["rr_sd"],
                "rr_stats_n": dm["nfh"]["rr_stats_n"],
            },
            "checkpoint_selection": "固定最后 epoch 导出编码器; 每 epoch 保存完整恢复状态",
        },
        "lp_resolved": {
            "data": "P0 隔离 train4809/val683 (SHA 见 data_manifest); 九类 multihot 原始顺序",
            "features": "共同整段编码器 8x64=512 维, eval+冻结",
            "head": "Linear(512,9); BCEWithLogitsLoss; Adam(lr1e-3,b(0.9,0.999),eps1e-8,wd0)",
            "batch_size": 128, "drop_last": False, "epochs_max": 100, "patience": 15,
            "selection": "严格最小元素加权 val BCE, 平局取更早 epoch",
            "head_seed": "head_seed = pretrain_seed",
            "no_tuning": "无增强/类别权重/scheduler/调参",
        },
        "r1_recipes": R1_RECIPES,
        "decision_rules": p["decision_rules"],
        "budgets": p["budgets"],
        "window": p["window"],
        "forbidden": p["forbidden"],
    }


def freeze(force: bool = False):
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    dst = C.RESULTS / "protocol_resolved.json"
    obj = build()
    if dst.exists() and not force:
        old = json.loads(dst.read_text(encoding="utf-8"))
        # git_head/dirty/frozen_at 允许漂移, 科学内容必须一致
        for k in ("pretrain_resolved", "lp_resolved", "r1_recipes", "decision_rules",
                  "budgets", "window", "paths", "impl_versions"):
            assert json.dumps(old[k], sort_keys=True) == json.dumps(obj[k], sort_keys=True), \
                f"protocol_resolved.json 科学内容被静默更改: {k}"
        print("protocol_resolved.json 已冻结且与当前实现一致")
        return dst
    dst.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print("protocol_resolved.json written (frozen):", dst)
    return dst


if __name__ == "__main__":
    freeze()
