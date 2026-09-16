import torch
import torch.nn as nn

from data_utils.augmentations import RandomLeadMask
from models.lead_attention import build_fusion
from models.vgg_1d import VGG16


def off_diagonal(matrix):
    rows, columns = matrix.shape
    if rows != columns:
        raise ValueError("cross-correlation matrix must be square")
    return matrix.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


def make_projector(input_dim, dimensions):
    sizes = [input_dim] + list(dimensions)
    layers = []
    for index in range(len(sizes) - 2):
        layers.extend(
            [
                nn.Linear(sizes[index], sizes[index + 1], bias=False),
                nn.BatchNorm1d(sizes[index + 1]),
                nn.ReLU(inplace=True),
            ]
        )
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
    return nn.Sequential(*layers)


class LeadFusionBT(nn.Module):
    def __init__(
        self,
        num_leads=8,
        projector=(2048, 2048, 2048),
        gamma=0.8,
        lambd=0.0051,
        fusion="concat",
        fusion_bt_weight=0.0,
        attention_hidden_dim=32,
        dropout=0.1,
        lead_mask_probability=0.0,
        min_masked_leads=1,
        max_masked_leads=2,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.feature_dim = 64
        self.gamma = gamma
        self.lambd = lambd
        self.fusion_name = fusion
        self.fusion_bt_weight = fusion_bt_weight

        self.backbone_group = nn.ModuleList()
        self.projector_group = nn.ModuleList()
        self.bn_group = nn.ModuleList()
        for _ in range(num_leads):
            backbone = VGG16(ch_in=1, alpha=0.125)
            backbone.fc = nn.Identity()
            self.backbone_group.append(backbone)
            self.projector_group.append(make_projector(self.feature_dim, projector))
            self.bn_group.append(nn.BatchNorm1d(projector[-1], affine=False))

        self.fusion = build_fusion(
            fusion,
            feature_dim=self.feature_dim,
            num_leads=num_leads,
            attention_hidden_dim=attention_hidden_dim,
            dropout=dropout,
        )
        if fusion_bt_weight > 0 and self.fusion is None:
            raise ValueError("fusion_bt_weight requires mean, eca, or attention fusion")
        if fusion_bt_weight > 0:
            self.fusion_projector = make_projector(self.feature_dim, (256, 256))
            self.fusion_bn = nn.BatchNorm1d(256, affine=False)
        else:
            self.fusion_projector = None
            self.fusion_bn = None
        self.lead_mask = RandomLeadMask(
            probability=lead_mask_probability,
            min_masked_leads=min_masked_leads,
            max_masked_leads=max_masked_leads,
        )

    def encode_leads(self, x):
        return torch.stack(
            [
                backbone(x[:, index : index + 1, :])
                for index, backbone in enumerate(self.backbone_group)
            ],
            dim=1,
        )

    def _barlow_loss(self, left, right, left_bn, right_bn):
        batch_size = left.shape[0]
        correlation = left_bn(left).T @ right_bn(right)
        correlation = correlation / batch_size
        on_diagonal = torch.diagonal(correlation).add(-1).pow(2).sum()
        off_diagonal_loss = off_diagonal(correlation).pow(2).sum()
        return on_diagonal + self.lambd * off_diagonal_loss

    def forward(self, view_a, view_b):
        if view_a.shape[0] < 2:
            raise ValueError("Barlow Twins requires a batch containing at least two samples")
        features_a = self.encode_leads(view_a)
        features_b = self.encode_leads(view_b)
        projections_a = [
            projector(features_a[:, index, :])
            for index, projector in enumerate(self.projector_group)
        ]
        projections_b = [
            projector(features_b[:, index, :])
            for index, projector in enumerate(self.projector_group)
        ]

        loss_intra = view_a.new_zeros(())
        loss_inter = view_a.new_zeros(())
        for left_index in range(self.num_leads):
            for right_index in range(self.num_leads):
                pair_loss = self._barlow_loss(
                    projections_a[left_index],
                    projections_b[right_index],
                    self.bn_group[left_index],
                    self.bn_group[right_index],
                )
                if left_index == right_index:
                    loss_intra = loss_intra + pair_loss
                else:
                    loss_inter = loss_inter + pair_loss
        loss_intra = loss_intra / self.num_leads
        loss_inter = loss_inter / (self.num_leads * (self.num_leads - 1))
        loss_lfbt = self.gamma * loss_intra + (1.0 - self.gamma) * loss_inter

        loss_fusion = view_a.new_zeros(())
        lead_weights = None
        if self.fusion_projector is not None:
            if self.lead_mask.probability > 0:
                masked_view_a, valid_a = self.lead_mask(view_a)
                masked_features_a = self.encode_leads(masked_view_a)
            else:
                valid_a = view_a.abs().sum(dim=-1) > 0
                masked_features_a = features_a
            valid_b = view_b.abs().sum(dim=-1) > 0
            fused_a, lead_weights = self.fusion(masked_features_a, valid_a)
            fused_b, _ = self.fusion(features_b, valid_b)
            projected_a = self.fusion_projector(fused_a)
            projected_b = self.fusion_projector(fused_b)
            loss_fusion = self._barlow_loss(
                projected_a,
                projected_b,
                self.fusion_bn,
                self.fusion_bn,
            )

        loss = loss_lfbt + self.fusion_bt_weight * loss_fusion
        return {
            "loss": loss,
            "loss_intra": loss_intra,
            "loss_inter": loss_inter,
            "loss_lfbt": loss_lfbt,
            "loss_fusion": loss_fusion,
            "lead_weights": lead_weights,
        }

    def encoder_state_dict_list(self):
        return [backbone.state_dict() for backbone in self.backbone_group]
