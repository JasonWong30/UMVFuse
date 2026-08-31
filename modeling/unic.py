import torch
import torch.nn as nn

from dinov2.models.image_standardVit import get_autoencoder


class FusionModel(nn.Module):
    def __init__(self, automodel: nn.Module):
        super().__init__()
        self.automodel = automodel
        embed_dim = self.automodel.encoder.embed_dim
        self.fusion_fc = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim, bias=False)

    def forward(
        self,
        ir_video: torch.Tensor,
        vi_video: torch.Tensor,
    ) -> torch.Tensor:
        vi_tokens, _, _, _ = self.automodel.forward_features(vi_video)
        ir_tokens, _, _, _ = self.automodel.forward_features(ir_video)

        concat_tokens = torch.cat([vi_tokens, ir_tokens], dim=-1)
        fused_tokens = self.fusion_fc(concat_tokens)
        fused_tokens = self.norm(fused_tokens)

        return self.automodel.forward_defeatures(fused_tokens, vi_video)


def build_fusionmodel_from_args(args):
    automodel = get_autoencoder('image_ae_base')

    if getattr(args, 'checkpoint', None):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        full_state_dict = checkpoint.get('model', checkpoint)
        full_state_dict = {
            key.removeprefix('module.'): value
            for key, value in full_state_dict.items()
        }

        automodel_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith('automodel.'):
                automodel_state_dict[key.removeprefix('automodel.')] = value
            elif key.startswith(('encoder.', 'decoder.')):
                automodel_state_dict[key] = value

        automodel.load_state_dict(automodel_state_dict, strict=True)

    return FusionModel(automodel)
