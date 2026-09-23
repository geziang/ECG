# -*- coding: utf-8 -*-
"""run_patient_probe.py — H3 配套诊断: 患者身份线性可解码度 (主机B, HOSTS §四 P1)。

问题: 冻结表征里能多大程度"认出患者"? 取预训练集中同患者 >=2 条记录的患者,
每患者取两条(锚/验), 以锚记录训练逻辑回归分类患者身份, 在验记录上测 top-1
准确率; chance = 1/患者数。B0 vs H3 checkpoint 的该值对比 = H3 是否真的
"去除"了患者身份信息的直接证据(与 Δ 配套报告)。

用法:
  python run_patient_probe.py --checkpoint checkpoint/ptxl_gamma08/encoder_group.pth \
      [--sinc-frontend 16] [--max-patients 400] [--out runlog/M/patient_probe_b0.json]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from models.vgg_1d import VGG16
from data_utils.seed_utils import set_seed


def build_encoders(checkpoint, sinc_m, device):
    load_params = torch.load(checkpoint, map_location=device, weights_only=True)
    encoders = []
    for i in range(8):
        enc = VGG16(ch_in=1, alpha=0.125)
        if sinc_m > 0:
            from models.sinc_conv import apply_sinc_frontend
            apply_sinc_frontend(enc, sinc_m)
        sd = load_params['backbone_state_dict_list'][i]
        enc.load_state_dict(sd, strict=False)
        enc.eval().to(device)
        encoders.append(enc)
    return encoders


@torch.no_grad()
def feature(encoders, x, device):
    """x: (8, T) numpy -> concat 8 导联 64 维嵌入 (512,)"""
    ts = torch.from_numpy(x.astype(np.float32)).to(device)
    feats = [enc(ts[i:i + 1, :]).squeeze(0).cpu().numpy() for i, enc in enumerate(encoders)]
    return np.concatenate(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, default=Path('data/pt_pretrain/samples'))
    ap.add_argument('--manifest', type=Path, default=Path('data/manifest.json'))
    ap.add_argument('--sinc-frontend', default=0, type=int)
    ap.add_argument('--max-patients', default=400, type=int)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoders = build_encoders(args.checkpoint, int(args.sinc_frontend), device)

    m = json.loads(args.manifest.read_text(encoding='utf-8'))
    by_patient = defaultdict(list)
    for r in m['records']:
        if r.get('in_pretrain'):
            by_patient[r['patient_id']].append(r['ecg_id'])
    multi = sorted([p for p, v in by_patient.items() if len(v) >= 2])
    rng = np.random.RandomState(0)
    rng.shuffle(multi)
    multi = multi[:args.max_patients]
    print(f'患者数(>=2条记录): {len(multi)}; chance = {1.0 / len(multi):.4f}')

    Xa, Xv = [], []   # 锚记录 / 验记录
    for p in multi:
        ids = sorted(by_patient[p])[:2]
        for bucket, ecg_id in ((Xa, ids[0]), (Xv, ids[1])):
            f = args.data_dir / f'sample_{ecg_id:05d}.npy'
            bucket.append(feature(encoders, np.load(f), device))
    Xa, Xv = np.stack(Xa), np.stack(Xv)

    scaler = StandardScaler().fit(Xa)
    clf = LogisticRegression(max_iter=500, C=1.0, multi_class='multinomial')
    clf.fit(scaler.transform(Xa), np.arange(len(multi)))
    acc = clf.score(scaler.transform(Xv), np.arange(len(multi)))
    res = dict(checkpoint=str(args.checkpoint), sinc_frontend=int(args.sinc_frontend),
               n_patients=len(multi), probe_top1_acc=round(float(acc), 4),
               chance=round(1.0 / len(multi), 4),
               lift=round(float(acc) - 1.0 / len(multi), 4))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
