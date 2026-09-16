"""Pre-registered physiological-spatial priors built on the B0 LFBT backbone.

The profiles in this file are scientific hypotheses, rather than a free
module-combination interface.  Every profile supports only its four planned
attribution ablations: baseline, physiology, spatial, and full.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ar_lfbt import LeadFusionBT, off_diagonal


LEAD_GROUPS = ((0, 1), (2, 3, 4, 5), (6, 7))
RESEARCH_PROFILES = {
    "physiospatial": {
        "physiology": "spectral",
        "spatial": "regional",
        "description": "Waveform-STFT morphology consistency plus anatomical lead-group consistency.",
    },
    "adaptive_spatial": {
        "physiology": "spectral",
        "spatial": "adaptive_relation",
        "description": "Waveform-STFT consistency plus view-stable data-driven lead relations.",
    },
    "multiscale_relation": {
        "physiology": "filterbank",
        "spatial": "relational_token",
        "description": "Sinc-initialized multi-band morphology alignment plus relation-aware lead-token consistency.",
    },
}
ABLATIONS = ("baseline", "physiology", "spatial", "full")
BASE_VARIANTS = ("b0", "ar_b3")


def base_variant_config(name):
    """Return the frozen B0 or previously selected AR-B3 training contract."""
    if name == "b0":
        return {
            "fusion": "concat",
            "fusion_bt_weight": 0.0,
            "lead_mask_probability": 0.0,
            "min_masked_leads": 1,
            "max_masked_leads": 2,
        }
    if name == "ar_b3":
        return {
            "fusion": "mean",
            "fusion_bt_weight": 0.2,
            "lead_mask_probability": 0.5,
            "min_masked_leads": 1,
            "max_masked_leads": 2,
        }
    raise ValueError(f"unknown base variant {name}; choose from {BASE_VARIANTS}")


class BarlowAlignment(nn.Module):
    def __init__(self, feature_dim=64, lambd=0.0051):
        super().__init__()
        self.lambd = lambd
        self.left_norm = nn.BatchNorm1d(feature_dim, affine=False)
        self.right_norm = nn.BatchNorm1d(feature_dim, affine=False)

    def forward(self, left, right):
        if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 2:
            raise ValueError("Barlow alignment expects matching tensors [B, D] with B >= 2")
        correlation = self.left_norm(left).T @ self.right_norm(right) / left.shape[0]
        return (
            torch.diagonal(correlation).add(-1).pow(2).sum()
            + self.lambd * off_diagonal(correlation).pow(2).sum()
        )


def log_spectrogram(waveform, n_fft=128, hop_length=32):
    """Create one log-magnitude STFT per lead: [B, L, F, Tau]."""
    if waveform.ndim != 3 or waveform.shape[-1] < n_fft:
        raise ValueError("waveform must have shape [B, L, T] with T >= n_fft")
    batch, leads, length = waveform.shape
    window = torch.hann_window(n_fft, dtype=waveform.dtype, device=waveform.device)
    spectrum = torch.stft(
        waveform.reshape(batch * leads, length),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    ).abs()
    return torch.log1p(spectrum).reshape(batch, leads, *spectrum.shape[1:])


class SpectralEncoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(32, feature_dim), nn.LayerNorm(feature_dim))

    def forward(self, spectra):
        return self.project(self.features(spectra))


class SpectralAlignmentPrior(nn.Module):
    """P1: align each lead's B0 waveform feature with its log-STFT feature."""

    def __init__(self, num_leads=8, feature_dim=64, lambd=0.0051):
        super().__init__()
        self.num_leads = num_leads
        self.encoder = SpectralEncoder(feature_dim)
        self.alignment = BarlowAlignment(feature_dim, lambd)

    def forward(self, waveform, lead_features):
        batch, leads = waveform.shape[:2]
        if leads != self.num_leads:
            raise ValueError("lead count does not match the configured prior")
        spectra = log_spectrogram(waveform)
        prior_features = self.encoder(spectra.reshape(batch * leads, 1, *spectra.shape[2:]))
        prior_features = prior_features.reshape(batch, leads, -1)
        return torch.stack([
            self.alignment(lead_features[:, lead], prior_features[:, lead])
            for lead in range(leads)
        ]).mean()


class LearnableFilterbank(nn.Module):
    """P2 frontend: ECG-scale sinc initialization, then modest end-to-end adaptation."""

    def __init__(self, bands=8, kernel_size=65):
        super().__init__()
        self.filters = nn.Conv1d(1, bands, kernel_size, padding=kernel_size // 2, bias=False)
        time = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        window = torch.hamming_window(kernel_size, periodic=False)
        edges = torch.linspace(0.01, 0.42, bands + 1)
        kernels = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            kernel = (2 * upper * torch.sinc(2 * upper * time) - 2 * lower * torch.sinc(2 * lower * time)) * window
            kernels.append(kernel / kernel.abs().sum().clamp_min(1e-6))
        with torch.no_grad():
            self.filters.weight.copy_(torch.stack(kernels).unsqueeze(1))

    def forward(self, waveform):
        return self.filters(waveform)


class FilterbankAlignmentPrior(nn.Module):
    def __init__(self, num_leads=8, feature_dim=64, lambd=0.0051):
        super().__init__()
        self.num_leads = num_leads
        self.filterbank = LearnableFilterbank()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(8), nn.GELU(), nn.AdaptiveAvgPool1d(16), nn.Flatten(),
            nn.Linear(128, feature_dim), nn.LayerNorm(feature_dim),
        )
        self.alignment = BarlowAlignment(feature_dim, lambd)

    def forward(self, waveform, lead_features):
        batch, leads, length = waveform.shape
        filtered = self.filterbank(waveform.reshape(batch * leads, 1, length))
        prior_features = self.encoder(filtered).reshape(batch, leads, -1)
        return torch.stack([
            self.alignment(lead_features[:, lead], prior_features[:, lead])
            for lead in range(leads)
        ]).mean()


class MaskedSpectralPredictionPrior(nn.Module):
    """P3: predict masked spectral summary while keeping an alignment path to B0."""

    def __init__(self, num_leads=8, feature_dim=64, lambd=0.0051, mask_ratio=0.3):
        super().__init__()
        self.num_leads = num_leads
        self.mask_ratio = mask_ratio
        self.encoder = SpectralEncoder(feature_dim)
        self.energy_head = nn.Sequential(nn.Linear(feature_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.alignment = BarlowAlignment(feature_dim, lambd)

    def forward(self, waveform, lead_features):
        spectra = log_spectrogram(waveform)
        mask = torch.rand_like(spectra) < self.mask_ratio
        batch, leads = spectra.shape[:2]
        prior_features = self.encoder(spectra.masked_fill(mask, 0.0).reshape(batch * leads, 1, *spectra.shape[2:]))
        prior_features = prior_features.reshape(batch, leads, -1)
        align_loss = torch.stack([
            self.alignment(lead_features[:, lead], prior_features[:, lead])
            for lead in range(leads)
        ]).mean()
        return align_loss + F.smooth_l1_loss(self.energy_head(prior_features).squeeze(-1), spectra.mean(dim=(-1, -2)))


class RegionalAlignmentPrior(nn.Module):
    """S1: soft consistency for limb, anterior, and lateral lead groups."""

    def __init__(self, num_leads=8, feature_dim=64, lambd=0.0051):
        super().__init__()
        if num_leads != 8:
            raise ValueError("regional alignment is defined for LFBT's eight leads")
        self.alignment = BarlowAlignment(feature_dim, lambd)

    def forward(self, left, right):
        return torch.stack([
            self.alignment(left[:, group, :].mean(dim=1), right[:, group, :].mean(dim=1))
            for group in LEAD_GROUPS
        ]).mean()


class AdaptiveRelationPrior(nn.Module):
    """S2: a data-driven relation matrix, used as a view-stability loss not a mixer."""

    def __init__(self, num_leads=8, feature_dim=64, temperature=0.2):
        super().__init__()
        self.num_leads = num_leads
        self.temperature = temperature
        self.project = nn.Linear(feature_dim, feature_dim, bias=False)
        self.lead_embedding = nn.Parameter(torch.empty(num_leads, feature_dim))
        nn.init.normal_(self.lead_embedding, std=0.02)

    def relation(self, features):
        features = F.normalize(self.project(features), dim=-1)
        data_scores = features @ features.transpose(-1, -2)
        lead_codes = F.normalize(self.lead_embedding, dim=-1)
        return torch.softmax((data_scores + lead_codes @ lead_codes.T) / self.temperature, dim=-1)

    def forward(self, left, right):
        relation_left, relation_right = self.relation(left), self.relation(right)
        # The weak off-diagonal term avoids a purely self-loop relation without imposing a fixed graph.
        identity = torch.eye(self.num_leads, device=left.device).unsqueeze(0)
        return F.mse_loss(relation_left, relation_right) + 0.01 * (relation_left * (1 - identity)).mean()


class RelationalTokenPrior(nn.Module):
    """S3: pairwise lead relations are aligned as an auxiliary signal only."""

    def __init__(self, num_leads=8, feature_dim=64, lambd=0.0051):
        super().__init__()
        self.num_leads = num_leads
        self.feature_dim = feature_dim
        self.query, self.key, self.value = (nn.Linear(feature_dim, feature_dim, bias=False) for _ in range(3))
        self.lead_embedding = nn.Parameter(torch.empty(num_leads, feature_dim))
        self.output = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.LayerNorm(feature_dim))
        self.alignment = BarlowAlignment(feature_dim, lambd)
        nn.init.normal_(self.lead_embedding, std=0.02)

    def _tokens(self, features):
        scores = self.query(features) @ self.key(features).transpose(-1, -2) / math.sqrt(self.feature_dim)
        scores = scores + (self.lead_embedding @ self.lead_embedding.T).unsqueeze(0)
        return self.output(torch.softmax(scores, dim=-1) @ self.value(features))

    def forward(self, left, right):
        left_tokens, right_tokens = self._tokens(left), self._tokens(right)
        return torch.stack([
            self.alignment(left_tokens[:, lead], right_tokens[:, lead])
            for lead in range(self.num_leads)
        ]).mean()


class PhysioSpatialLFBT(LeadFusionBT):
    """B0 or AR-B3 LFBT plus exactly one pre-registered physiology-space hypothesis."""

    def __init__(
        self,
        profile="physiospatial",
        base_variant="b0",
        ablation="full",
        physiology_weight=0.05,
        spatial_weight=0.05,
        **kwargs,
    ):
        if profile not in RESEARCH_PROFILES:
            raise ValueError(f"unknown profile {profile}; choose from {tuple(RESEARCH_PROFILES)}")
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation {ablation}; choose from {ABLATIONS}")
        base_config = base_variant_config(base_variant)
        conflicting = set(kwargs).intersection(base_config)
        if conflicting:
            raise ValueError(f"base_variant owns these settings: {sorted(conflicting)}")
        super().__init__(**base_config, **kwargs)
        self.profile, self.base_variant, self.ablation = profile, base_variant, ablation
        self.physiology_weight, self.spatial_weight = physiology_weight, spatial_weight
        configuration = RESEARCH_PROFILES[profile]
        prior_kwargs = {"num_leads": self.num_leads, "feature_dim": self.feature_dim, "lambd": self.lambd}
        self.physiology_prior = {
            "spectral": SpectralAlignmentPrior,
            "filterbank": FilterbankAlignmentPrior,
            "masked_spectral": MaskedSpectralPredictionPrior,
        }[configuration["physiology"]](**prior_kwargs)
        spatial_classes = {
            "regional": RegionalAlignmentPrior,
            "adaptive_relation": AdaptiveRelationPrior,
            "relational_token": RelationalTokenPrior,
        }
        if configuration["spatial"] == "adaptive_relation":
            self.spatial_prior = AdaptiveRelationPrior(num_leads=self.num_leads, feature_dim=self.feature_dim)
        else:
            self.spatial_prior = spatial_classes[configuration["spatial"]](**prior_kwargs)

    def forward(self, view_a, view_b):
        if view_a.shape[0] < 2:
            raise ValueError("Barlow Twins requires a batch containing at least two samples")
        features_a, features_b = self.encode_leads(view_a), self.encode_leads(view_b)
        projections_a = [projector(features_a[:, index]) for index, projector in enumerate(self.projector_group)]
        projections_b = [projector(features_b[:, index]) for index, projector in enumerate(self.projector_group)]
        loss_intra, loss_inter = view_a.new_zeros(()), view_a.new_zeros(())
        for left_index in range(self.num_leads):
            for right_index in range(self.num_leads):
                pair_loss = self._barlow_loss(projections_a[left_index], projections_b[right_index], self.bn_group[left_index], self.bn_group[right_index])
                if left_index == right_index:
                    loss_intra = loss_intra + pair_loss
                else:
                    loss_inter = loss_inter + pair_loss
        loss_intra = loss_intra / self.num_leads
        loss_inter = loss_inter / (self.num_leads * (self.num_leads - 1))
        loss_lfbt = self.gamma * loss_intra + (1 - self.gamma) * loss_inter
        loss_fusion = view_a.new_zeros(())
        if self.fusion_projector is not None:
            if self.lead_mask.probability > 0:
                masked_view_a, valid_a = self.lead_mask(view_a)
                masked_features_a = self.encode_leads(masked_view_a)
            else:
                valid_a = view_a.abs().sum(dim=-1) > 0
                masked_features_a = features_a
            valid_b = view_b.abs().sum(dim=-1) > 0
            fused_a, _ = self.fusion(masked_features_a, valid_a)
            fused_b, _ = self.fusion(features_b, valid_b)
            loss_fusion = self._barlow_loss(
                self.fusion_projector(fused_a),
                self.fusion_projector(fused_b),
                self.fusion_bn,
                self.fusion_bn,
            )
        use_physiology = self.ablation in ("physiology", "full")
        use_spatial = self.ablation in ("spatial", "full")
        loss_physiology = view_a.new_zeros(())
        loss_spatial = view_a.new_zeros(())
        if use_physiology:
            loss_physiology = 0.5 * (self.physiology_prior(view_a, features_a) + self.physiology_prior(view_b, features_b))
        if use_spatial:
            loss_spatial = self.spatial_prior(features_a, features_b)
        loss_base = loss_lfbt + self.fusion_bt_weight * loss_fusion
        loss = loss_base + self.physiology_weight * loss_physiology + self.spatial_weight * loss_spatial
        return {
            "loss": loss,
            "loss_intra": loss_intra,
            "loss_inter": loss_inter,
            "loss_lfbt": loss_lfbt,
            "loss_fusion": loss_fusion,
            "loss_base": loss_base,
            "loss_physiology": loss_physiology,
            "loss_spatial": loss_spatial,
        }
