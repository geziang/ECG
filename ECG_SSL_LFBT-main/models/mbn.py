from pathlib import Path

import torch
import torch.nn as nn

from models.lead_attention import build_fusion
from models.vgg_1d import VGG16
from utils.checkpoint import load_torch_checkpoint


def _checkpoint_state_lists(checkpoint):
    if "backbone_state_dict_list" in checkpoint:
        return checkpoint["backbone_state_dict_list"]
    if "backbone_state_dict" in checkpoint:
        states = checkpoint["backbone_state_dict"]
        return [item.state_dict() if hasattr(item, "state_dict") else item for item in states]
    return None


class MultiBranchNet(nn.Module):
    def __init__(
        self,
        num_classes,
        num_leads=8,
        checkpoint=None,
        fusion="concat",
        feature_dim=64,
        attention_hidden_dim=32,
        dropout=0.1,
        map_location="cpu",
        blur_pool=0,
        pool_power=0.0,
        trc=0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_leads = num_leads
        self.feature_dim = feature_dim
        self.fusion_name = fusion
        self.encoder_group = nn.ModuleList()
        for _ in range(num_leads):
            backbone = VGG16(ch_in=1, alpha=0.125, blur_pool=int(blur_pool),
                               pool_power=float(pool_power), trc=int(trc))
            if backbone.output_dim != feature_dim:
                raise ValueError("feature_dim does not match the VGG scaling factor")
            self.encoder_group.append(backbone.model)

        self.fusion = build_fusion(
            fusion,
            feature_dim=feature_dim,
            num_leads=num_leads,
            attention_hidden_dim=attention_hidden_dim,
            dropout=dropout,
        )
        final_dim = num_leads * feature_dim
        if self.fusion is not None:
            final_dim += feature_dim
        self.fc = nn.Linear(final_dim, num_classes)

        if checkpoint is not None and str(checkpoint).lower() != "none":
            self.load_pretrained(checkpoint, map_location=map_location)

    def load_pretrained(self, checkpoint_path, map_location="cpu"):
        checkpoint = load_torch_checkpoint(Path(checkpoint_path), map_location=map_location)
        state_list = _checkpoint_state_lists(checkpoint)
        if state_list is None:
            raise ValueError(f"checkpoint has no encoder state list: {checkpoint_path}")
        if len(state_list) != self.num_leads:
            raise ValueError(
                f"checkpoint contains {len(state_list)} encoders, expected {self.num_leads}"
            )
        for index, (encoder, state) in enumerate(zip(self.encoder_group, state_list)):
            model_state = {
                key.removeprefix("model."): value
                for key, value in state.items()
                if key.startswith("model.")
            }
            if not model_state:
                model_state = state
            missing, unexpected = encoder.load_state_dict(model_state, strict=False)
            if unexpected or missing:
                raise ValueError(
                    f"lead {index} checkpoint mismatch: missing={missing}, unexpected={unexpected}"
                )
        fusion_state = checkpoint.get("fusion_state_dict")
        if self.fusion is not None and fusion_state:
            saved_fusion = checkpoint.get("fusion", checkpoint.get("config", {}).get("fusion"))
            if saved_fusion and saved_fusion != self.fusion_name:
                raise ValueError(
                    f"checkpoint fusion is {saved_fusion}, requested {self.fusion_name}"
                )
            self.fusion.load_state_dict(fusion_state)

    def extract_lead_features(self, x):
        if x.ndim != 3 or x.shape[1] != self.num_leads:
            raise ValueError(f"expected [B, {self.num_leads}, L], received {tuple(x.shape)}")
        features = [
            encoder(x[:, lead_index : lead_index + 1, :]).flatten(1)
            for lead_index, encoder in enumerate(self.encoder_group)
        ]
        return torch.stack(features, dim=1)

    def forward_features(self, x, valid_mask=None):
        lead_features = self.extract_lead_features(x)
        if valid_mask is None:
            valid_mask = x.abs().sum(dim=-1) > 0
        lead_features = lead_features * valid_mask.to(lead_features.dtype).unsqueeze(-1)
        original_feature = lead_features.flatten(1)
        fusion_feature = None
        lead_weights = None
        final_feature = original_feature
        if self.fusion is not None:
            fusion_feature, lead_weights = self.fusion(lead_features, valid_mask)
            final_feature = torch.cat([original_feature, fusion_feature], dim=1)
        return {
            "lead_features": lead_features,
            "original_feature": original_feature,
            "fusion_feature": fusion_feature,
            "lead_weights": lead_weights,
            "final_feature": final_feature,
        }

    def forward(self, x, valid_mask=None, return_features=False):
        features = self.forward_features(x, valid_mask=valid_mask)
        logits = self.fc(features["final_feature"])
        if return_features:
            features["logits"] = logits
            return features
        return logits

    def freeze_encoders(self):
        for encoder in self.encoder_group:
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    def freeze_fusion(self):
        if self.fusion is None:
            return
        self.fusion.eval()
        for parameter in self.fusion.parameters():
            parameter.requires_grad = False
