import torch
import torch.nn as nn


class FusionModel(nn.Module):

    def __init__(self, automodel: nn.Module):
        super().__init__()
        self.automodel = automodel
        embed_dim = self.automodel.encoder.embed_dim
        self.fusion_fc = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim, bias=False)

    def forward(self, ir_video: torch.Tensor, vi_video: torch.Tensor) -> torch.Tensor:
        (vi_tokens, _, _, _) = self.automodel.forward_features(vi_video)
        (ir_tokens, _, _, _) = self.automodel.forward_features(ir_video)
        concat_tokens = torch.cat([vi_tokens, ir_tokens], dim=-1)
        fused_tokens = self.fusion_fc(concat_tokens)
        fused_tokens = self.norm(fused_tokens)
        fused_video = self.automodel.forward_defeatures(fused_tokens, vi_video)
        return fused_video
