"""C1 链收尾: 读取 c1_nfh_b0 LP metrics.json, 查重后落 matrix csv 行。幂等可重跑。"""
import datetime
import json
import sys

m = json.load(open(r'F:\新实验\ECG_SSL_LFBT-main\feat\M_c1_nfh_b0_seed0\metrics.json'))
csv = r'F:\新实验\ECG_SSL_LFBT-main\runlog\M\matrix_results.csv'
rows = open(csv, encoding='utf-8').read()
if 'c1_nfh_b0,' in rows:
    print('CSV_EXISTS_SKIP')
    sys.exit(0)
delta = round(m['auprc'] - 0.7177, 4)
ts = datetime.datetime.now().strftime('%m-%d %H:%M')
with open(csv, 'a', encoding='utf-8') as f:
    f.write(f"{ts},c1_nfh_b0,0,--data-dir data/pt_pretrain_nfh --epochs 100,"
            f"{m['auroc']:.4f},{m['auprc']:.4f},{delta},0\n")
print('CSV_APPENDED auprc=%.4f delta=%.4f' % (m['auprc'], delta))
