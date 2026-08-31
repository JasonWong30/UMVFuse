from functools import partial
import torch
import torch.nn as nn
from timm.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp


class PatchEmbed(nn.Module):
    """Image to Patch Embedding (2D)"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0

        self.img_size = img_size
        self.patch_size = patch_size

        self.h_tiles = img_size[0] // patch_size[0]
        self.w_tiles = img_size[1] // patch_size[1]
        self.num_patches = self.h_tiles * self.w_tiles

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, N, C]
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x, y):
        B, N, C = x.shape
        q = (
            self.q(y)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)

        return x

class CrossBlock(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=CrossAttention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, y):
        x = self.drop_path(self.attn(self.norm1(x), self.norm1(y)))
        return x

class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ImageViTEncoder(nn.Module):
    def __init__(
        self,
        img_size=384,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        no_qkv_bias=False,
        qk_scale=None,
        drop_path_rate=0.0,
        dropout=0.5,
        norm_layer=nn.LayerNorm,
        cls_embed=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_embed = cls_embed

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        if cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_len = num_patches + 1
        else:
            pos_len = num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, pos_len, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                    attn_func=partial(
                        Attention,
                        input_size=(1, self.patch_embed.h_tiles, self.patch_embed.w_tiles),
                    ),
                )
                for i in range(depth)
            ]
        )


        self.norm = norm_layer(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, 2)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if cls_embed:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: [B, C, H, W]
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, D]

        if self.cls_embed:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        
        # if self.cls_embed:
        cls_token = x[:, 0, :]  # [B, D]
        feature_tokens = x[:, 1:, :]  # [B, N, D]
        
        # Produce binary classification logits with the classification head.
        cls_output = self.dropout(cls_token)
        logits = self.head(cls_output)  # [B, 2]
        
        return logits, feature_tokens, cls_token

        # return x  # [B, N, D]


class ImageViTDecoder(nn.Module):
    def __init__(
        self,
        img_size=384,
        patch_size=16,
        in_chans=3,
        encoder_embed_dim=768,
        decoder_embed_dim=512,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans

        self.h_tiles = self.img_size[0] // self.patch_size[0]
        self.w_tiles = self.img_size[1] // self.patch_size[1]
        self.num_patches = self.h_tiles * self.w_tiles

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))

        self.blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    drop_path=0.0,
                    attn_func=partial(
                        Attention,
                        input_size=(1, self.h_tiles, self.w_tiles),
                    ),
                )
                for _ in range(depth)
            ]
        )

        self.norm = norm_layer(decoder_embed_dim)
        self.pred = nn.Linear(
            decoder_embed_dim,
            self.patch_size[0] * self.patch_size[1] * in_chans,
            bias=True,
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.conv = nn.Conv2d(3, 3, 9, 1, 4) 

    def unpatchify(self, x):
        # x: [B, N, P*P*C]
        B, N, PPc = x.shape
        p_h, p_w = self.patch_size
        C = self.in_chans
        assert N == self.num_patches
        H = self.h_tiles * p_h
        W = self.w_tiles * p_w

        x = x.view(B, self.h_tiles, self.w_tiles, p_h, p_w, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, H_tiles, p_h, W_tiles, p_w]
        x = x.view(B, C, H, W)
        return x

    def forward(self, tokens):
        # tokens: [B, N, D_enc]
        x = self.decoder_embed(tokens)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.pred(x)  # [B, N, P*P*C]
        img = self.unpatchify(x)

        return img + self.conv(img)


class ImageAutoencoderViT(nn.Module):
    def __init__(
        self,
        num_frames=6,
        img_size=384,
        patch_size=16,
        in_chans=3,
        encoder_embed_dim=768,
        encoder_depth=6,
        encoder_heads=12,
        encoder_mlp_ratio=4.0,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_heads=8,
        decoder_mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        cls_embed=True,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.encoder_depth = encoder_depth

        self.encoder = ImageViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=encoder_mlp_ratio,
            cls_embed=cls_embed,
        )

        self.decoder = ImageViTDecoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=decoder_mlp_ratio,
            norm_layer=norm_layer,
        )

    def forward_features(self, video):
        
        Bv, C, T, H, W = video.shape
        assert H == self.img_size[0] and W == self.img_size[1]

        frames = video.permute(0, 2, 1, 3, 4).contiguous().view(Bv * T, C, H, W)
        # logits, feature_tokens, cls_token = self.encoder(frames)  # [B*T, N, D]

        B = frames.shape[0]
        x = self.encoder.patch_embed(frames)  # [B, N, D]

        if self.encoder.cls_embed:
            cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
            
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.encoder.pos_embed

        output_patch = [x[:, 1:, :]]  
        # for i, blk in enumerate(self.encoder.blocks):
        for i, blk  in enumerate (self.encoder.blocks):
            x = blk(x)                                           
            output_patch.append(x[:, 1:, :]) 

        x = self.encoder.norm(x)
        
        # if self.cls_embed:
        cls_token = x[:, 0, :]  # [B, D]
        cls_token_video = cls_token.view(Bv, T, -1).mean(1)

        feature_tokens = x[:, 1:, :]  # [B, N, D]
        
        # Produce binary classification logits with the classification head.
        cls_output = self.encoder.dropout(cls_token_video)
        logits = self.encoder.head(cls_output)  # [B, 2]

        return feature_tokens, cls_token, output_patch, logits

    def forward_defeatures(self, tokens, image):
        B, C, T, H, W = image.shape

        recon_imgs = self.decoder(tokens)  # [B*T, C, H, W]
        recon_video = recon_imgs.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        return recon_video
        
    def forward(self, video):
        # video: [B, C, T, H, W]
        B, C, T, H, W = video.shape
        assert H == self.img_size[0] and W == self.img_size[1]

        frames = video.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        logits, feature_tokens, cls_token = self.encoder(frames)  # [B*T, N, D]

        recon_imgs = self.decoder(feature_tokens)  # [B*T, C, H, W]
        recon_video = recon_imgs.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        return recon_video, logits, cls_token

def image_ae_base(**kwargs):
    kwargs.setdefault("img_size", 384)
    kwargs.setdefault("patch_size", 16)
    kwargs.setdefault("in_chans", 3)
    model = ImageAutoencoderViT(
        encoder_embed_dim=768,
        encoder_depth=6,
        encoder_heads=12,
        encoder_mlp_ratio=4.0,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_heads=8,
        decoder_mlp_ratio=4.0,
        **kwargs,
    )
    return model


def get_autoencoder(arch="image_ae_base", **kwargs):
    model = globals().get(arch)(**kwargs)
    return model
