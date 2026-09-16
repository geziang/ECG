import torch
import torch.nn as nn


def _validate_features(features, valid_mask):
    if features.ndim != 3:
        raise ValueError("lead features must have shape [B, D, F]")
    if valid_mask is None:
        valid_mask = torch.ones(
            features.shape[:2], dtype=torch.bool, device=features.device
        )
    if valid_mask.shape != features.shape[:2]:
        raise ValueError("valid_mask must have shape [B, D]")
    if (~valid_mask).all(dim=1).any():
        raise ValueError("each sample must retain at least one valid lead")
    return valid_mask


class MeanLeadFusion(nn.Module):
    def forward(self, features, valid_mask=None):
        valid_mask = _validate_features(features, valid_mask)
        weights = valid_mask.to(features.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        fused = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return fused, weights


class ECALeadFusion(nn.Module):
    def __init__(self, num_leads=8, kernel_size=3):
        super().__init__()
        del num_leads
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, features, valid_mask=None):
        valid_mask = _validate_features(features, valid_mask)
        descriptor = features.mean(dim=-1).unsqueeze(1)
        scores = torch.sigmoid(self.conv(descriptor)).squeeze(1)
        scores = scores * valid_mask.to(scores.dtype)
        weights = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
        fused = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return fused, weights


class LeadAttentionFusion(nn.Module):
    def __init__(self, feature_dim=64, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.proj_h = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.proj_context = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, valid_mask=None):
        valid_mask = _validate_features(features, valid_mask)
        valid = valid_mask.to(features.dtype).unsqueeze(-1)
        context = (features * valid).sum(dim=1, keepdim=True)
        context = context / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        hidden = torch.tanh(self.proj_h(features) + self.proj_context(context))
        scores = self.score(self.dropout(hidden)).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        fused = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return fused, weights


def build_fusion(
    fusion,
    feature_dim=64,
    num_leads=8,
    attention_hidden_dim=32,
    dropout=0.1,
):
    if fusion == "mean":
        return MeanLeadFusion()
    if fusion == "eca":
        return ECALeadFusion(num_leads=num_leads)
    if fusion == "attention":
        return LeadAttentionFusion(
            feature_dim=feature_dim,
            hidden_dim=attention_hidden_dim,
            dropout=dropout,
        )
    if fusion == "concat":
        return None
    raise ValueError(f"unsupported fusion: {fusion}")
