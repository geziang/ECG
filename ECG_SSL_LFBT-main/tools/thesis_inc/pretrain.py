# -*- coding: utf-8 -*-
"""thesis-inc 预训练入口: 单作业 PT(含完整断点恢复/截止控制/资源记账)。"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # ECG_SSL_LFBT-main

from tools.thesis_inc import common as C
from tools.thesis_inc.pt_dataset import NFHPTDataset
from tools.thesis_inc.pt_model import IncBT, init_fingerprint

BATCH_SIZE = 128
WORKERS = 4
PRINT_FREQ = 100
LR = 0.001


class DeadlineStop(Exception):
    """达到 GPU 停止截止(10-26 11:30)或预算上限时抛出; 状态已保存。"""


def recipe_needs(recipe: dict):
    template = None
    local = None
    if recipe.get("template_mode"):
        template = recipe["template_mode"]
    if recipe.get("local_mode"):
        local = recipe["local_mode"]
    return template, local, recipe.get("rr_weight", 0) > 0


def make_dataset(cfg: dict, epoch: int, limit=None) -> NFHPTDataset:
    template, local, need_rr = recipe_needs(cfg["recipe"])
    ds = NFHPTDataset(
        samples_dir=C.CODE_ROOT / "data" / "pt_pretrain_nfh" / "samples",
        rpeak_npz=C.CODE_ROOT / "data" / "pt_rpeaks_nfh.npz",
        seed=cfg["seed"], template_mode=template, local_mode=local,
        need_rr=need_rr, limit=limit)
    ds.epoch = epoch
    return ds


def epoch_perm_batches(seed: int, epoch: int, n: int, bs: int):
    g = torch.Generator()
    g.manual_seed(int.from_bytes(
        hashlib.blake2b(f"perm|{seed}|{epoch}".encode(), digest_size=8).digest(), "big")
        % (2 ** 63))
    perm = torch.randperm(n, generator=g).tolist()
    return [perm[i:i + bs] for i in range(0, n - bs + 1, bs)]  # drop_last=True


def full_rng_state() -> dict:
    return {"py": random.getstate(), "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(st: dict):
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch"].cpu().to(torch.uint8)
                        if hasattr(st["torch"], "cpu") else st["torch"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8)
                                      if hasattr(s, "cpu") else s for s in st["cuda"]])


def _to_cpu(sd):
    if isinstance(sd, list):
        return [{k: v.cpu() if torch.is_tensor(v) else v for k, v in s.items()} for s in sd]
    return {k: v.cpu() if torch.is_tensor(v) else v for k, v in sd.items()}


def save_train_state(path: Path, epoch: int, step_in_epoch: int, global_step: int,
                     model: IncBT, optimizer, elapsed_s: float, cfg_sha: str,
                     aux_counters: dict):
    snap = {}
    for name in ("backbone_group", "projector_group", "bn_group", "rr_head"):
        mod = getattr(model, name, None)
        if mod is None:
            continue
        snap[name] = _to_cpu([m.state_dict() for m in mod] if isinstance(mod, torch.nn.ModuleList)
                             else mod.state_dict())
    opt = optimizer.state_dict()
    opt["state"] = {k: {k2: (v2.cpu() if torch.is_tensor(v2) else v2)
                        for k2, v2 in v.items()} for k, v in opt["state"].items()}
    st = {"epoch": epoch, "step_in_epoch": step_in_epoch, "global_step": global_step,
          "model": snap, "optimizer": opt, "rng": full_rng_state(),
          "elapsed_s": elapsed_s, "config_sha": cfg_sha,
          "aux_counters": aux_counters, "saved_at": C.bj_str()}
    tmp = path.with_suffix(".tmp")
    torch.save(st, tmp)
    tmp.replace(path)


def run_pt_job(cfg: dict, stop_deadline, budget_guard=None, limit=None) -> dict:
    """cfg: {job_id, stage, recipe_name, seed, epochs, recipe{...}}; 返回结果 dict。

    recipe 字段: template_mode|local_mode, morph_weight, rr_weight。
    """
    job_id = cfg["job_id"]
    job_dir = C.JOBS / job_id
    ckpt_dir = C.CKPT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    done = job_dir / "DONE.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))

    # 科学身份哈希: 不含 epochs(进度目标; R5 允许显式 20->100 续训), 续训不得改其余字段
    cfg_sha = hashlib.sha256(json.dumps(
        {k: cfg[k] for k in ("job_id", "stage", "recipe_name", "seed", "recipe")},
        sort_keys=True).encode()).hexdigest()
    (job_dir / "config.json").write_text(
        json.dumps({**cfg, "config_sha": cfg_sha}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    log_f = open(job_dir / "train_log.jsonl", "a", encoding="utf-8")

    def log(**kw):
        rec = {"ts": C.bj_str(), "job_id": job_id, **kw}
        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_f.flush()

    t0 = time.time()
    last_flush = t0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C.set_seed(cfg["seed"])
    template, local, need_rr = recipe_needs(cfg["recipe"])
    model = IncBT(use_rr=need_rr).to(dev)
    fp = init_fingerprint(model)
    parameters = model.param_groups_pb()
    optimizer = torch.optim.Adam(parameters, lr=LR, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    log(event="init", init_fingerprint=fp, n_params=sum(p.numel() for p in parameters))

    ds = make_dataset(cfg, epoch=0, limit=limit)
    n = len(ds)
    log(event="dataset", n_records=n, rr_mu=ds.rr_mu.tolist(), rr_sd=ds.rr_sd.tolist(),
        rr_n_valid=ds.n_valid_rr, aug_impl=ds.aug_impl_version)

    state_path = ckpt_dir / "train_state.pth"
    start_epoch, start_step, global_step, elapsed = 0, 0, 0, 0.0
    aux_counters = {"local_skip_batches": 0, "valid_records_seen": 0, "records_seen": 0}
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu")
        assert st["config_sha"] == cfg_sha, "resume 配置不一致"
        for name, sd in st["model"].items():
            mod = getattr(model, name)
            if isinstance(mod, torch.nn.ModuleList):
                for m, s in zip(mod, sd):
                    m.load_state_dict(s)
            else:
                mod.load_state_dict(sd)
        optimizer.load_state_dict(st["optimizer"])
        restore_rng_state(st["rng"])
        start_epoch = st["epoch"] + (1 if st["step_in_epoch"] == 0 else 0)
        start_step = 0 if st["step_in_epoch"] == 0 else st["step_in_epoch"]
        if st["step_in_epoch"] == 0:
            start_epoch = st["epoch"] + 1
        else:  # epoch 内中断: 本 epoch 从中断 step 继续
            start_epoch = st["epoch"]
        global_step = st["global_step"]
        elapsed = st.get("elapsed_s", 0.0)
        aux_counters.update(st.get("aux_counters", {}))
        log(event="resume", from_epoch=start_epoch, step_in_epoch=start_step)

    epochs = cfg["epochs"]
    morph_w = float(cfg["recipe"].get("morph_weight", 0))
    rr_w = float(cfg["recipe"].get("rr_weight", 0))
    status = "trained"
    stop_reason = "all_epochs"
    epoch, step = start_epoch, start_step

    def flush_budget(force=False):
        nonlocal last_flush
        now = time.time()
        if force or now - last_flush > 900:
            C.add_resource("gpu", now - last_flush, job_id)
            last_flush = now

    try:
        for epoch in range(start_epoch, epochs):
            ds.epoch = epoch
            batches = epoch_perm_batches(cfg["seed"], epoch, n, BATCH_SIZE)
            loader = torch.utils.data.DataLoader(
                ds, batch_sampler=batches, num_workers=WORKERS,
                pin_memory=True, persistent_workers=False)
            model.train()
            ep_losses = {"loss": 0.0, "loss_r": 0.0, "loss_t": 0.0, "aux": 0.0}
            n_steps = 0
            ep_t0 = time.time()
            for step, batch in enumerate(loader):
                if step < start_step:  # epoch 内断点恢复: 跳过已做步(批内容确定性)
                    continue
                if step % 50 == 0:
                    now = C.now_bj()
                    if now >= stop_deadline:
                        raise DeadlineStop(f"gpu_stop deadline {stop_deadline.isoformat()}")
                    if budget_guard is not None and not budget_guard():
                        raise DeadlineStop("budget guard trip")
                y1 = batch["v1"].to(dev, non_blocking=True)
                y2 = batch["v2"].to(dev, non_blocking=True)
                valid = batch["valid"].to(dev, non_blocking=True)
                if need_rr or template:
                    z1_list, f1_list = model._embed_feat(y1)
                    z2_list, _ = model._embed_feat(y2)
                else:
                    z1_list = model._embed(y1)
                    z2_list = model._embed(y2)
                    f1_list = None
                loss_bt, loss_r, loss_t = model.bt_loss(z1_list, z2_list, BATCH_SIZE)
                loss = loss_bt
                aux_val = None
                diag_mean = None
                if template is not None and morph_w > 0:
                    tpl = batch["template"].to(dev, non_blocking=True)
                    aux_val = model.template_loss(z1_list, z2_list, tpl, valid)
                    loss = loss + morph_w * aux_val
                elif local is not None and morph_w > 0:
                    w1 = batch["w1"].to(dev, non_blocking=True)
                    w2 = batch["w2"].to(dev, non_blocking=True)
                    wmask = batch["wmask"].to(dev, non_blocking=True)
                    aux_val, diag_mean = model.local_loss(w1, w2, wmask)
                    if diag_mean is None:
                        aux_counters["local_skip_batches"] += 1
                    loss = loss + morph_w * aux_val
                if rr_w > 0:
                    rr3 = batch["rr3"].to(dev, non_blocking=True)
                    l_rr = model.rr_loss(f1_list, rr3, valid)
                    loss = loss + rr_w * l_rr
                aux_counters["valid_records_seen"] += int((valid > 0.5).sum().item())
                aux_counters["records_seen"] += int(valid.numel())

                grad_ratio = None
                if (template is not None or local is not None) and morph_w > 0 \
                        and aux_val is not None and step % (PRINT_FREQ * 5) == 0:
                    # 辅助/主干梯度范数比(lead0 backbone 探针, backward 前计算, 图仍保留)
                    try:
                        probe = list(model.backbone_group[0].parameters())
                        g_bt = torch.autograd.grad(loss_bt, probe, retain_graph=True,
                                                   allow_unused=True)
                        g_aux = torch.autograd.grad(morph_w * aux_val, probe,
                                                    retain_graph=True, allow_unused=True)
                        n_bt = torch.sqrt(sum(g.pow(2).sum() for g in g_bt if g is not None))
                        n_aux = torch.sqrt(sum(g.pow(2).sum() for g in g_aux if g is not None))
                        grad_ratio = (n_aux / n_bt.clamp(min=1e-12)).item()
                    except Exception:
                        grad_ratio = None

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                n_steps += 1
                ep_losses["loss"] += loss.item()
                ep_losses["loss_r"] += loss_r.item()
                ep_losses["loss_t"] += loss_t.item()
                ep_losses["aux"] += float(aux_val.item()) if aux_val is not None else 0.0
                if step % PRINT_FREQ == 0:
                    rec = {"epoch": epoch, "step": step,
                           "loss": round(loss.item(), 5),
                           "loss_r": round(loss_r.item(), 5),
                           "loss_t": round(loss_t.item(), 5)}
                    if aux_val is not None:
                        rec["aux"] = round(aux_val.item(), 6)
                    if diag_mean is not None:
                        rec["local_diag_mean"] = round(float(diag_mean), 5)
                    if grad_ratio is not None:
                        rec["grad_aux_bt"] = round(grad_ratio, 5)
                    log(**rec)
                flush_budget()
            ep_wall = time.time() - ep_t0
            k = max(n_steps, 1)
            log(event="epoch_end", epoch=epoch, wall_s=round(ep_wall, 1),
                loss=round(ep_losses["loss"] / k, 5),
                loss_r=round(ep_losses["loss_r"] / k, 5),
                loss_t=round(ep_losses["loss_t"] / k, 5),
                aux=round(ep_losses["aux"] / k, 6),
                local_skip_batches=aux_counters["local_skip_batches"],
                elapsed_s=round(elapsed + time.time() - t0, 1))
            save_train_state(state_path, epoch, 0, global_step, model, optimizer,
                             elapsed + time.time() - t0, cfg_sha, aux_counters)
            start_step = 0
    except DeadlineStop as e:
        save_train_state(state_path, epoch, step, global_step, model, optimizer,
                         elapsed + time.time() - t0, cfg_sha, aux_counters)
        status, stop_reason = "partial", f"deadline_stop:{e}"
        log(event="deadline_stop", epoch=epoch, step=step, reason=str(e))
    finally:
        flush_budget(force=True)
        wall = time.time() - t0

    result = {"job_id": job_id, "stage": cfg["stage"], "kind": "pretrain",
              "recipe": cfg["recipe_name"], "seed": cfg["seed"],
              "epochs_target": epochs, "status": status, "stop_reason": stop_reason,
              "init_fingerprint": fp, "wall_s": round(wall, 1),
              "global_step": global_step, "aux_counters": aux_counters,
              "config_sha": cfg_sha, "git_head": C.git_head()}
    if status == "trained":
        enc_path = ckpt_dir / "encoder_group.pth"
        torch.save({"backbone_state_dict_list":
                    [m.state_dict() for m in model.backbone_group],
                    "backbone_state_dict":
                    [m.state_dict() for m in model.backbone_group]},
                   enc_path)
        result["encoder_sha256"] = C.sha256_of(enc_path)
        result["encoder_path"] = str(enc_path)
        log(event="export_encoder", sha=result["encoder_sha256"])
    (job_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    if status == "trained":
        done.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    C.append_csv(C.RESULTS / "results.csv", {
        "job_id": job_id, "stage": cfg["stage"], "kind": "pretrain",
        "recipe": cfg["recipe_name"], "seed": cfg["seed"],
        "pt_epochs": epochs, "status": status, "stop_reason": stop_reason,
        "wall_s": result["wall_s"], "macro_auprc": "", "macro_auroc": "",
        "encoder_sha256": result.get("encoder_sha256", "")})
    log_f.close()
    return result
