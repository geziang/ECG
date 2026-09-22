import time
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from data_utils.cls_datasets import get_data_loaders
from data_utils.seed_utils import set_seed
from models.mbn import MultiBranchNet
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

parser = argparse.ArgumentParser(description='Lead-Fusion Barlow Twins Fine Tuning')
parser.add_argument('--data-dir', type=str, required=True,
                    metavar='DIR', help='data path')
parser.add_argument('--checkpoint', type=Path, default=None, nargs='?',
                    metavar='DIR', help='path to checkpoint (传 none/省略 = TFS 随机初始化对照)')
parser.add_argument('--num-classes', type=int, required=True, metavar='N', help="the number of classes")
parser.add_argument('--fraction', default=1.0, type=float, metavar='L',
                    help='The fraction of training set used for training.')
parser.add_argument('--model-dir', default='./', type=Path,
                    metavar='DIR', help='path to save models')
parser.add_argument('--workers', default=6, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.0001, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--seed', default=0, type=int, metavar='N', help='random seed')
# ===== 与预训练一致的架构开关(任务书 08 §3.2: FT 必须传递并校验, 否则 C2 FT10 不可审计) =====
parser.add_argument('--trc', default=0, type=int, help='TRC/GRN1D: 与预训练一致')
parser.add_argument('--blur-pool', default=0, type=int, help='H2: 与预训练一致')
parser.add_argument('--pool-power', default=0.0, type=float, help='T3: 与预训练一致')
# ===== W3 B-2 评估扩展(主机B, 默认全关 -> W2 行为与 metrics.json 内容不变) =====
parser.add_argument('--extended-metrics', default=0, type=int,
                    help='B-2: 1=metrics.json 附加 per-class AP/Macro-F1/Sens/Spec/ECE/Brier')
parser.add_argument('--save-predictions', default='', type=str,
                    help='B-2: 逐记录 y_true/y_pred/y_prob 落盘目录(必须位于 runlog/W3/ 下; 空=不保存)')
parser.add_argument('--protocol-id', default='', type=str, help='B-2: 协议标识元数据')
parser.add_argument('--data-manifest-sha', default='', type=str, help='B-2: 数据 manifest SHA256 元数据')
parser.add_argument('--checkpoint-sha', default='', type=str, help='B-2: checkpoint SHA256 元数据(留空且有 checkpoint 时自动计算)')


class FineTuning(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # 架构一致性校验(任务书 08 §3.2): checkpoint 旁 config.json 记录的架构开关必须与 FT 参数一致
        if args.checkpoint is not None and str(args.checkpoint).lower() != "none":
            _cfg_p = Path(args.checkpoint).parent / "config.json"
            if _cfg_p.exists():
                _cfg = json.loads(_cfg_p.read_text(encoding="utf-8"))
                _a = _cfg.get("args", {})
                for _k, _v in (("trc", int(args.trc)), ("blur_pool", int(args.blur_pool)),
                               ("pool_power", float(args.pool_power))):
                    try:
                        _ck = float(_a.get(_k, 0))
                    except (TypeError, ValueError):
                        _ck = 0.0
                    if abs(_ck - float(_v)) > 1e-9:
                        raise SystemExit(
                            f"[arch-mismatch] checkpoint 训练时 {_k}={_a.get(_k)} 但 FT 传 {_v};"
                            f" FT 架构参数必须与预训练一致 (config: {_cfg_p})")
        self.model = MultiBranchNet(args.num_classes, checkpoint=args.checkpoint,
                                    blur_pool=int(args.blur_pool),
                                    pool_power=float(args.pool_power),
                                    trc=int(args.trc)).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def train(self, data_loader, optimizer):
        self.model.train()
        for batch_idx, (data, target) in tqdm(enumerate(data_loader)):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            optimizer.step()

    def validate(self, data_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in tqdm(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()  # sum up batch loss
        val_loss /= len(data_loader)
        print('\nVal loss: {:.4f}\n'.format(val_loss))
        return val_loss

    def test(self, data_loader):
        self.model.eval()
        y_pred_list = list()
        y_true_list = list()
        y_pred_s_list = list()
        with torch.no_grad():
            for data, target in tqdm(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                output_softmax = nn.functional.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                y_true_list.append(target.cpu().numpy())
                y_pred_list.append(pred.cpu().numpy())
                y_pred_s_list.append(output_softmax.cpu().numpy())
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        y_pred_s = np.concatenate(y_pred_s_list, axis=0)
        y_true_s = np.eye(len(np.unique(y_true)))[y_true]
        auroc = roc_auc_score(y_true_s, y_pred_s, average="macro")
        auprc = average_precision_score(y_true_s, y_pred_s)
        conf_mat = confusion_matrix(y_true, y_pred)
        self.last_metrics = (auroc, auprc, conf_mat)
        self.last_pred = (y_true, y_pred, y_pred_s)  # W3 B-2: 暂存逐记录, 供可选扩展使用
        print("Test Performance -----------------------------------------")
        print("AUROC: ", auroc)
        print("AUPRC: ", auprc)
        print("Confusion Matrix: \n", conf_mat)

    def run(self):
        train_loader, val_loader, test_loader = get_data_loaders(self.args.data_dir,
                                                                 self.args.batch_size,
                                                                 self.args.workers,
                                                                 train_ratio=self.args.fraction)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        min_loss = 999999999
        self.args.model_dir.mkdir(parents=True, exist_ok=True)
        best_path = self.args.model_dir / "ft_best_ckpt.pth"
        for ep in range(1, self.args.epochs + 1):
            print("\nEpoch %d ----------------------------" % ep)
            self.train(train_loader, opt)
            v_loss = self.validate(val_loader)
            if v_loss < min_loss:
                min_loss = v_loss
                print("Save checkpoint with minimum val_loss (%f)." % v_loss)
                torch.save(self.model.state_dict(), best_path)
        # 用 map_location 加载, 兼容 CPU/不同 GPU (指南 §5 P1)
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        self.test(test_loader)
        auroc, auprc, conf_mat = self.last_metrics
        metrics = dict(auroc=float(auroc), auprc=float(auprc), confusion_matrix=conf_mat.tolist(),
                       fraction=self.args.fraction, seed=self.args.seed,
                       checkpoint=str(self.args.checkpoint) if self.args.checkpoint else "random-init")

        # ===== W3 B-2 扩展(默认关闭: 不加参数时 metrics.json 与 W2 逐字节一致) =====
        if int(getattr(self.args, 'extended_metrics', 0)) or str(getattr(self.args, 'save_predictions', '') or ''):
            import metrics_ext as _mx
            _y_true, _y_pred, _y_prob = self.last_pred
            _meta = {
                "protocol_id": str(getattr(self.args, 'protocol_id', '') or ''),
                "data_manifest_sha": str(getattr(self.args, 'data_manifest_sha', '') or ''),
                "checkpoint": str(self.args.checkpoint) if self.args.checkpoint else "random-init",
                "checkpoint_sha256": (str(getattr(self.args, 'checkpoint_sha', '') or '')
                                      or (_mx.sha256_of(self.args.checkpoint)
                                          if self.args.checkpoint else "")),
                "eval": "ft", "fraction": self.args.fraction,
                "num_classes": int(_y_prob.shape[1]), "seed": self.args.seed,
                "code_sha": "see git",
            }
            if int(getattr(self.args, 'extended_metrics', 0)):
                metrics.update(_mx.compute_all(_y_true, _y_prob))
                print("Extended metrics: macro_f1=%.4f ece=%.4f brier=%.4f"
                      % (metrics["macro_f1"], metrics["ece"], metrics["brier_multiclass"]))
            _sp_dir = str(getattr(self.args, 'save_predictions', '') or '')
            if _sp_dir:
                _mx.save_eval_artifacts(_sp_dir, _y_true, _y_pred, _y_prob, _meta)
                print("Per-record predictions saved to", _sp_dir)

        with open(self.args.model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=1)


def wait_for_gpu_memory(need_mib=2500, poll_s=60):
    """启动显存闸门(2026-09-17): LP/FT 为轻任务, 仅在空闲显存极低时排队, 防止挤爆并发预训练。"""
    import subprocess as _sp
    import time as _time
    while True:
        try:
            out = _sp.run(['nvidia-smi', '--query-gpu=memory.free',
                           '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, timeout=60).stdout
            free = int(out.strip().splitlines()[0])
        except Exception:
            return
        if free >= need_mib:
            return
        print(f"[vram-gate] 等待显存: 需要 {need_mib}MiB, 当前空闲 {free}MiB ({poll_s}s 后重查)", flush=True)
        _time.sleep(poll_s)


def main():
    args = parser.parse_args()
    if args.checkpoint is not None and str(args.checkpoint).lower() == "none":
        args.checkpoint = None  # TFS 随机初始化对照
    wait_for_gpu_memory()
    set_seed(args.seed)
    print("Fine Tuning Setting ======================================")
    print(args)
    print("==========================================================")
    fine_tuning = FineTuning(args)
    fine_tuning.run()


if __name__ == '__main__':
    main()
