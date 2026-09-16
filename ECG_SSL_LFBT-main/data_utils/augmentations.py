from dataclasses import dataclass
import random
import numpy as np
import torch
from scipy.signal import resample


def rrc(sig, crop_ratio_low, crop_ratio_up):
    sig_length = sig.shape[1]
    crop_length = int(random.uniform(crop_ratio_low, crop_ratio_up) * sig_length)
    if sig_length == crop_length:
        start_point = 0
    elif crop_length == 0:
        return sig
    else:
        start_point = np.random.randint(0, sig_length - crop_length)
    end_point = start_point + crop_length
    sig_crop = sig[:, start_point:end_point]
    sig_crop = resample(sig_crop, sig_length, axis=1)
    return sig_crop


def to(sig, mask_ratio_low, mask_ratio_up):
    num_leads = sig.shape[0]
    sig_length = sig.shape[1]
    mask_length = int(random.uniform(mask_ratio_low, mask_ratio_up) * sig_length)
    mask_start = np.random.randint(0, sig_length - mask_length)
    if mask_length > 0:
        # 先拷贝再置零, 避免原地修改输入数组 (指南 §5 P2)
        sig = np.array(sig, copy=True)
        sig[:, mask_start:(mask_start + mask_length)] = np.zeros([num_leads, mask_length])
    return sig


def rrc_to(sig, crop_ratio_low=0.5, crop_ratio_up=1.0, mask_ratio_low=0.0, mask_ratio_up=0.5):
    return to(rrc(sig, crop_ratio_low, crop_ratio_up), mask_ratio_low, mask_ratio_up)


class RandomResizeCropTimeOut(object):
    def __init__(self, params=None):
        if params is None:
            self.params = [0.5, 1.0, 0.0, 0.5]
        else:
            self.params = params

    def __call__(self, x):
        return rrc_to(x, self.params[0], self.params[1], self.params[2], self.params[3])


class ToTensor(object):
    def __init__(self, leads=None):
        self.leads = leads
        pass

    def __call__(self, x):
        if self.leads is None:
            return torch.tensor(x).float()
        else:
            return torch.tensor(x[self.leads, :]).float()
        return torch.as_tensor(np.array(signal, copy=True), dtype=torch.float32)


@dataclass
class RandomLeadMask:
    probability: float = 0.5
    min_masked_leads: int = 1
    max_masked_leads: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if self.min_masked_leads < 0:
            raise ValueError("min_masked_leads must be non-negative")
        if self.max_masked_leads < self.min_masked_leads:
            raise ValueError("max_masked_leads must be >= min_masked_leads")

    def __call__(self, signal):
        if torch.is_tensor(signal):
            return self._mask_tensor(signal)
        return self._mask_array(np.asarray(signal))

    def _mask_tensor(self, signal: torch.Tensor):
        squeeze_batch = signal.ndim == 2
        batched = signal.unsqueeze(0) if squeeze_batch else signal
        if batched.ndim != 3:
            raise ValueError("expected [D, L] or [B, D, L]")
        output = batched.clone()
        batch_size, num_leads, _ = output.shape
        valid_mask = torch.ones(batch_size, num_leads, dtype=torch.bool, device=output.device)
        max_masked = min(self.max_masked_leads, max(0, num_leads - 1))
        min_masked = min(self.min_masked_leads, max_masked)
        for batch_index in range(batch_size):
            if torch.rand((), device=output.device).item() >= self.probability or max_masked == 0:
                continue
            count = int(torch.randint(min_masked, max_masked + 1, (), device=output.device).item())
            if count == 0:
                continue
            indices = torch.randperm(num_leads, device=output.device)[:count]
            output[batch_index, indices, :] = 0
            valid_mask[batch_index, indices] = False
        if squeeze_batch:
            return output[0], valid_mask[0]
        return output, valid_mask

    def _mask_array(self, signal: np.ndarray):
        squeeze_batch = signal.ndim == 2
        batched = signal[None, ...] if squeeze_batch else signal
        if batched.ndim != 3:
            raise ValueError("expected [D, L] or [B, D, L]")
        output = np.array(batched, copy=True)
        batch_size, num_leads, _ = output.shape
        valid_mask = np.ones((batch_size, num_leads), dtype=bool)
        max_masked = min(self.max_masked_leads, max(0, num_leads - 1))
        min_masked = min(self.min_masked_leads, max_masked)
        for batch_index in range(batch_size):
            if np.random.random() >= self.probability or max_masked == 0:
                continue
            count = np.random.randint(min_masked, max_masked + 1) if max_masked else 0
            indices = np.random.choice(num_leads, size=count, replace=False)
            output[batch_index, indices, :] = 0
            valid_mask[batch_index, indices] = False
        if squeeze_batch:
            return output[0], valid_mask[0]
        return output, valid_mask


class SpeedPerturbRRC_TO(object):
    """速度扰动 + RRC-TO(语音 speed perturb 迁移): 先随机时间缩放(模拟心率变化),
    再走原版 rrc_to。speed=(1.0,1.0) 时与 RandomResizeCropTimeOut 行为一致。"""

    def __init__(self, params=None, speed=(1.0, 1.0)):
        self.params = params if params is not None else [0.5, 1.0, 0.0, 0.5]
        self.speed = speed

    def __call__(self, x):
        orig_len = x.shape[1]
        lo, hi = self.speed
        if lo < 1.0 or hi > 1.0:
            factor = random.uniform(lo, hi)
            x = resample(x, max(32, int(orig_len / factor)), axis=1)
        out = rrc_to(x, *self.params)
        if out.shape[1] != orig_len:
            out = resample(out, orig_len, axis=1)
        return out


class AdjacentLeadSwap(object):
    """相邻导联互换增强(电极错位模拟, 阵列信号迁移): 以 prob 概率随机交换一对相邻胸导。
    对接 37 条件电极错位鲁棒评估线。prob=0 时恒等。"""

    CHEST_PAIRS = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]  # (V1,V2)...(V5,V6)

    def __init__(self, prob=0.0):
        self.prob = prob

    def __call__(self, x):
        if self.prob <= 0 or random.random() >= self.prob:
            return x
        i, j = random.choice(self.CHEST_PAIRS)
        x = np.array(x, copy=True)
        x[[i, j], :] = x[[j, i], :]
        return x
