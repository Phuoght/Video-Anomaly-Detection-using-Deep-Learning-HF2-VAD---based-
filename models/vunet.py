# Revised VUnet + Swin-like blocks (with per-encoder linear drop-path scheduling)
# - Adds: shifted-window attention mask, padding for non-divisible H/W,
#   relative positional bias, safe sampling (dtype/device), DropPath improvements,
#   mask caching, optional use of SDPA when possible, and small init fixes.
# - Each SwinEncoder now accepts `drop_path_rate` (max) and applies linear decay across its layers.
# - Keep external dependencies from `models.basic_modules` as in original.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict
import numpy as np
from torch.nn import ModuleDict, ModuleList, Conv2d

# keep your imports from original code (assumed present in project)
from models.basic_modules import (
    VUnetResnetBlock,
    Upsample,
    Downsample,
    NormConv2d,
    SpaceToDepth,
    DepthToSpace,
)

# ------------------------
# Utilities
# ------------------------

def retrieve(config, key, default=None):
    keys = key.split('/')
    value = config
    for k in keys:
        value = value.get(k, default)
        if value is default:
            return default
    return value


class DropPath(nn.Module):
    """Stochastic Depth per sample.
    Implementation using Bernoulli to avoid in-place scaling issues.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = (random_tensor < keep_prob).to(x.dtype)
        x = x * binary_mask / keep_prob
        return x


# ------------------------
# Window partition / reverse with padding support
# ------------------------

def _get_padding_hw(H: int, W: int, window_size: int) -> Tuple[int, int]:
    pad_h = (math.ceil(H / window_size) * window_size) - H
    pad_w = (math.ceil(W / window_size) * window_size) - W
    return pad_h, pad_w


def window_partition(x: Tensor, window_size: int) -> Tensor:
    """
    x: (B, C, H, W) -> windows: (num_windows*B, window_size, window_size, C)
    This function assumes input is already padded so H,W divisible by window_size.
    """
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, "H/W must be divisible by window_size"
    x = x.reshape(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1)  # (B, Hws, Wws, ws, ws, C)
    windows = x.reshape(-1, window_size, window_size, C)  # (num_windows*B, ws, ws, C)
    return windows


def window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
    """
    windows: (num_windows*B, ws, ws, C) -> x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H // window_size * W // window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.reshape(B, -1, H, W)
    return x


# ------------------------
# Relative positional bias helper
# ------------------------
class RelativePositionBias(nn.Module):
    def __init__(self, window_size: int, num_heads: int):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        # table size (2*ws-1)*(2*ws-1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        # get pair-wise relative position index for each token inside window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, ws, ws
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, L, L
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # L, L, 2
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # L, L
        self.register_buffer('relative_position_index', relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self) -> Tensor:
        # returns (num_heads, L, L)
        values = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        values = values.reshape(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        values = values.permute(2, 0, 1).contiguous()
        return values


# ------------------------
# Attention (supports mask + relative bias)
# ------------------------
class MHSASDPA(nn.Module):
    """Multi-Head Self-Attention with optional mask and relative positional bias.
    Two execution paths:
      - fast-path: use F.scaled_dot_product_attention when no mask and no relative bias
      - general-path: compute attention manually to support mask and bias
    Input x: (B_windows, L, C)
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, window_size: Optional[int] = None):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.window_size = window_size
        self.rel_pos = RelativePositionBias(window_size, num_heads) if window_size is not None else None

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        # x: (B_, L, C)
        B_, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to (B_, num_heads, L, head_dim)
        q = q.reshape(B_, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B_, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B_, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Fast path: no mask and no rel_pos -> use SDPA
        if attn_mask is None and self.rel_pos is None:
            # F.scaled_dot_product_attention accepts (B, heads, L, head_dim) on recent torch
            try:
                attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
                attn_out = attn_out.permute(0, 2, 1, 3).reshape(B_, L, C)
                x = self.proj(attn_out)
                x = self.proj_drop(x)
                return x
            except Exception:
                # fallback to manual computation if environment doesn't support that signature
                pass

        # General path: compute manually to support mask and relative bias
        # q,k: (B, heads, L, d)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, L, L)

        if self.rel_pos is not None:
            # rel_pos returns (num_heads, L, L) -> broadcast to (B, heads, L, L)
            rel_bias = self.rel_pos()
            attn = attn + rel_bias.unsqueeze(0)

        if attn_mask is not None:
            # attn_mask expected boolean where True indicates position to be masked
            # Expand attn_mask to (B, heads, L, L) if currently (num_windows, L, L)
            if attn_mask.dtype == torch.bool:
                if attn_mask.shape[0] == attn.shape[0]:
                    mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
                elif attn_mask.shape[0] == 1:
                    mask = attn_mask.unsqueeze(1).repeat(attn.shape[0], self.num_heads, 1, 1)
                else:
                    # try broadcasting
                    mask = attn_mask
                mask = mask.to(attn.device)
                attn = attn.masked_fill(mask, float('-inf'))
            else:
                # if float mask additive
                attn = attn + attn_mask

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # (B, heads, L, d)
        out = out.permute(0, 2, 1, 3).reshape(B_, L, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ------------------------
# MLP
# ------------------------
class SwinMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 4)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ------------------------
# Swin Block
# ------------------------
class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 8,
        shift: bool = False,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        layerscale_init: float = 1e-6,
        use_sdpa: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift = shift
        self.use_sdpa = use_sdpa

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMLP(dim, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma1 = nn.Parameter(layerscale_init * torch.ones(dim))
        self.gamma2 = nn.Parameter(layerscale_init * torch.ones(dim))

        self.attn = MHSASDPA(dim=dim, num_heads=num_heads, dropout=dropout, window_size=window_size)

        # cache masks for given spatial shapes to avoid recomputation
        self._mask_cache: Dict[Tuple[int, int, int, int], Tensor] = {}

    def _get_attn_mask(self, H: int, W: int, ws: int, shift_size: int, device: torch.device) -> Optional[Tensor]:
        if shift_size == 0:
            return None
        key = (H, W, ws, shift_size)
        if key in self._mask_cache:
            return self._mask_cache[key]

        # create img_mask labeling different regions following Swin implementation
        img_mask = torch.zeros((1, 1, H, W), device=device, dtype=torch.int)
        h_slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)  # (num_windows, ws, ws, 1)
        mask_windows = mask_windows.reshape(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) != mask_windows.unsqueeze(2)  # (num_windows, L, L)
        self._mask_cache[key] = attn_mask
        return attn_mask

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        ws = self.window_size
        # compute padding if needed
        pad_h, pad_w = _get_padding_hw(H, W, ws)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H_pad, W_pad = H + pad_h, W + pad_w
        else:
            H_pad, W_pad = H, W

        actual_ws = min(ws, H_pad, W_pad)
        shift_size = actual_ws // 2 if self.shift and actual_ws > 1 else 0

        # optional cyclic shift
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))

        # partition
        windows = window_partition(x, actual_ws)  # (num_win*B, ws, ws, C)
        nw = windows.shape[0]
        windows = windows.reshape(nw, actual_ws * actual_ws, C)  # (nw, L, C)

        # pre-norm
        windows_norm = self.norm1(windows)

        # compute appropriate attn_mask for shifted windows
        attn_mask = self._get_attn_mask(H_pad, W_pad, actual_ws, shift_size, x.device) if shift_size > 0 else None

        # attention
        attn_out = self.attn(windows_norm, attn_mask=attn_mask)

        # residual + layerscale + droppath
        out1 = windows + self.drop_path(self.gamma1 * attn_out)
        out2 = out1 + self.drop_path(self.gamma2 * self.mlp(self.norm2(out1)))

        # restore windows -> (B, C, H_pad, W_pad)
        out2 = out2.reshape(nw, actual_ws, actual_ws, C)
        x = window_reverse(out2, actual_ws, H_pad, W_pad)

        # reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))

        # remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x


# ------------------------
# Swin Encoder (stacked blocks) with per-layer linear drop-path schedule
# ------------------------
class SwinEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        window_size: int = 8,
        shift: bool = True,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        layerscale_init: float = 1e-6,
        use_sdpa: bool = True,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            do_shift = shift and (i % 2 == 1)
            # linear schedule across the layers of this encoder (0 -> drop_path_rate)
            dp = drop_path_rate * i / max(1, num_layers - 1)
            layers.append(
                SwinBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift=do_shift,
                    dropout=dropout,
                    drop_path=dp,
                    layerscale_init=layerscale_init,
                    use_sdpa=use_sdpa,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ------------------------
# Transformer wrapper blocks (NoSkip and WithSkip)
# ------------------------
class TransformerBlockNoSkip(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        window_size: int = 8,
        shift: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float = 1e-6,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.swin = SwinEncoder(
            dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            window_size=window_size,
            shift=shift,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            layerscale_init=layerscale_init,
            use_sdpa=use_sdpa,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.swin(x)
        B, C, H, W = out.shape
        out_flat = out.reshape(B, C, -1).permute(0, 2, 1)  # (B, L, C)
        out_flat = self.norm(out_flat)
        out = out_flat.permute(0, 2, 1).reshape(B, C, H, W)
        return out


class TransformerBlockWithSkip(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        window_size: int = 8,
        shift: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float = 1e-6,
        use_sdpa: bool = True,
    ):
        super().__init__()
        # Here the original code used embed_dim*2 for concat of skip
        self.swin = SwinEncoder(
            dim=embed_dim * 2,
            num_layers=num_layers,
            num_heads=num_heads,
            window_size=window_size,
            shift=shift,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            layerscale_init=layerscale_init,
            use_sdpa=use_sdpa,
        )
        self.proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, h: Tensor, skip: Tensor) -> Tensor:
        x = torch.cat([h, skip], dim=1)
        x = self.swin(x)
        x = self.proj(x)
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, -1).permute(0, 2, 1)
        x_flat = self.norm(x_flat)
        x = x_flat.permute(0, 2, 1).reshape(B, C, H, W)
        return x


# ------------------------
# VUnet encoder/decoder/bottleneck and full model
# ------------------------
class VUnetEncoder(nn.Module):
    def __init__(
        self,
        n_stages,
        nf_in=3,
        nf_start=128,
        nf_max=256,
        n_rnb=2,
        conv_layer=NormConv2d,
        dropout_prob=0.2,
    ):
        super().__init__()
        self.in_op = conv_layer(nf_in, nf_start, kernel_size=1)
        nf = nf_start
        self.blocks = ModuleDict()
        self.downs = ModuleDict()
        self.n_rnb = n_rnb
        self.n_stages = n_stages
        for i_s in range(self.n_stages):
            if i_s > 0:
                self.downs.update(
                    {
                        f"s{i_s + 1}": Downsample(
                            nf, min(2 * nf, nf_max), conv_layer=conv_layer
                        )
                    }
                )
                nf = min(2 * nf, nf_max)
            for ir in range(self.n_rnb):
                stage = f"s{i_s + 1}_{ir + 1}"
                self.blocks.update(
                    {
                        stage: VUnetResnetBlock(
                            nf, conv_layer=conv_layer, dropout_prob=dropout_prob
                        )
                    }
                )

    def forward(self, x):
        out = {}
        h = self.in_op(x)
        for ir in range(self.n_rnb):
            h = self.blocks[f"s1_{ir + 1}"](h)
            out[f"s1_{ir + 1}"] = h
        for i_s in range(1, self.n_stages):
            h = self.downs[f"s{i_s + 1}"](h)
            for ir in range(self.n_rnb):
                stage = f"s{i_s + 1}_{ir + 1}"
                h = self.blocks[stage](h)
                out[stage] = h
        return out


class ZConverter(nn.Module):
    def __init__(self, n_stages, nf, device, conv_layer=NormConv2d, dropout_prob=0.2, drop_path_rate=0.0):
        super().__init__()
        self.n_stages = n_stages
        self.device = device
        # pass drop_path_rate down to transformer blocks
        self.blocks = ModuleList([
            TransformerBlockWithSkip(embed_dim=nf, dropout=dropout_prob, drop_path_rate=drop_path_rate) for _ in range(3)
        ])
        self.conv1x1 = conv_layer(nf, nf, 1)
        self.up = Upsample(in_channels=nf, out_channels=nf, conv_layer=conv_layer)
        self.channel_norm = conv_layer(2 * nf, nf, 1)
        self.d2s = DepthToSpace(block_size=2)
        self.s2d = SpaceToDepth(block_size=2)

    def forward(self, x_f):
        params = {}
        zs = {}
        h = self.conv1x1(x_f[f"s{self.n_stages}_2"])
        for n, i_s in enumerate(range(self.n_stages, self.n_stages - 2, -1)):
            stage = f"s{i_s}"
            spatial_size = x_f[stage + "_2"].shape[-1]
            spatial_stage = "%dby%d" % (spatial_size, spatial_size)
            h = self.blocks[2 * n](h, x_f[stage + "_2"])
            params[spatial_stage] = h
            z = self._latent_sample(params[spatial_stage])
            zs[spatial_stage] = z
            if n == 0:
                gz = torch.cat([x_f[stage + "_1"], z], dim=1)
                gz = self.channel_norm(gz)
                h = self.blocks[2 * n + 1](h, gz)
                h = self.up(h)
        return params, zs

    def _latent_sample(self, mean: Tensor) -> Tensor:
        # ensure dtype/device matching
        sample = mean.new_empty(mean.size()).normal_()
        return mean + sample


class VUnetDecoder(nn.Module):
    def __init__(
        self,
        n_stages,
        nf=128,
        nf_out=3,
        n_rnb=2,
        conv_layer=NormConv2d,
        spatial_size=256,
        final_act=True,
        dropout_prob=0.2,
    ):
        super().__init__()
        self.final_act = final_act
        self.blocks = ModuleDict()
        self.ups = ModuleDict()
        self.n_stages = n_stages
        self.n_rnb = n_rnb
        for i_s in range(self.n_stages - 2, 0, -1):
            if i_s == 1:
                self.ups.update(
                    {
                        f"s{i_s + 1}": Upsample(
                            in_channels=nf, out_channels=nf // 2, conv_layer=conv_layer,
                        )
                    }
                )
                nf = nf // 2
            else:
                self.ups.update(
                    {
                        f"s{i_s + 1}": Upsample(
                            in_channels=nf, out_channels=nf, conv_layer=conv_layer,
                        )
                    }
                )
            for ir in range(self.n_rnb, 0, -1):
                stage = f"s{i_s}_{ir}"
                self.blocks.update(
                    {
                        stage: VUnetResnetBlock(
                            nf,
                            use_skip=True,
                            conv_layer=conv_layer,
                            dropout_prob=dropout_prob,
                        )
                    }
                )
        self.final_layer = conv_layer(nf, nf_out, kernel_size=1)
        self.final_act = nn.Sigmoid()

    def forward(self, x, skips):
        out = x
        for i_s in range(self.n_stages - 2, 0, -1):
            out = self.ups[f"s{i_s + 1}"](out)
            for ir in range(self.n_rnb, 0, -1):
                stage = f"s{i_s}_{ir}"
                out = self.blocks[stage](out, skips[stage])
        out = self.final_layer(out)
        if self.final_act:
            out = self.final_act(out)
        return out


class VUnetBottleneck(nn.Module):
    def __init__(
        self,
        n_stages,
        nf,
        device,
        n_rnb=2,
        n_auto_groups=4,
        conv_layer=NormConv2d,
        dropout_prob=0.2,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.device = device
        self.blocks = ModuleDict()
        self.channel_norm = ModuleDict()
        self.conv1x1 = conv_layer(nf, nf, 1)
        self.up = Upsample(in_channels=nf, out_channels=nf, conv_layer=conv_layer)
        self.depth_to_space = DepthToSpace(block_size=2)
        self.space_to_depth = SpaceToDepth(block_size=2)
        self.n_stages = n_stages
        self.n_rnb = n_rnb
        self.n_auto_groups = n_auto_groups
        for i_s in range(self.n_stages, self.n_stages - 2, -1):
            self.channel_norm.update({f"s{i_s}": conv_layer(2 * nf, nf, 1)})
            for ir in range(self.n_rnb):
                self.blocks.update(
                    {
                        f"s{i_s}_{ir + 1}": TransformerBlockWithSkip(
                            embed_dim=nf, dropout=dropout_prob, drop_path_rate=drop_path_rate
                        )
                    }
                )
        self.auto_blocks = ModuleList([
            TransformerBlockNoSkip(embed_dim=nf, dropout=dropout_prob, drop_path_rate=drop_path_rate)
        ])
        for _ in range(3):
            self.auto_blocks.append(
                TransformerBlockWithSkip(embed_dim=nf, dropout=dropout_prob, drop_path_rate=drop_path_rate)
            )
        self.param_converter = conv_layer(4 * nf, nf, kernel_size=1)

    def forward(self, x_e, z_post):
        p_params = {}
        z_prior = {}
        use_z = True
        h = self.conv1x1(x_e[f"s{self.n_stages}_2"])
        for i_s in range(self.n_stages, self.n_stages - 2, -1):
            stage = f"s{i_s}"
            spatial_size = x_e[stage + "_2"].shape[-1]
            spatial_stage = "%dby%d" % (spatial_size, spatial_size)
            h = self.blocks[stage + "_2"](h, x_e[stage + "_2"])
            if spatial_size == 1:
                p_params[spatial_stage] = h
                prior_samples = self._latent_sample(p_params[spatial_stage])
                z_prior[spatial_stage] = prior_samples
            else:
                if use_z:
                    z_flat = (
                        self.space_to_depth(z_post[spatial_stage])
                        if z_post[spatial_stage].shape[2] > 1
                        else z_post[spatial_stage]
                    )
                    sec_size = z_flat.shape[1] // 4
                    z_groups = torch.split(
                        z_flat, [sec_size, sec_size, sec_size, sec_size], dim=1
                    )
                param_groups = []
                sample_groups = []
                param_features = self.auto_blocks[0](h)
                param_features = self.space_to_depth(param_features)
                param_features = self.param_converter(param_features)
                for i_a in range(len(self.auto_blocks)):
                    param_groups.append(param_features)
                    prior_samples = self._latent_sample(param_groups[-1])
                    sample_groups.append(prior_samples)
                    if i_a + 1 < len(self.auto_blocks):
                        feedback = z_groups[i_a] if use_z else prior_samples
                        param_features = self.auto_blocks[i_a + 1](param_features, feedback)
                p_params_stage = self.__merge_groups(param_groups)
                prior_samples = self.__merge_groups(sample_groups)
                p_params[spatial_stage] = p_params_stage
                z_prior[spatial_stage] = prior_samples
            if use_z:
                z = (
                    self.depth_to_space(z_post[spatial_stage])
                    if z_post[spatial_stage].shape[-1] != h.shape[-1]
                    else z_post[spatial_stage]
                )
            else:
                z = prior_samples
            gz = torch.cat([x_e[stage + "_1"], z], dim=1)
            gz = self.channel_norm[stage](gz)
            h = self.blocks[stage + "_1"](h, gz)
            if i_s == self.n_stages:
                h = self.up(h)
        return h, p_params, z_prior

    def __split_groups(self, x):
        sec_size = x.shape[1] // 4
        return torch.split(
            self.space_to_depth(x), [sec_size, sec_size, sec_size, sec_size], dim=1,
        )

    def __merge_groups(self, x):
        return self.depth_to_space(torch.cat(x, dim=1))

    def _latent_sample(self, mean: Tensor) -> Tensor:
        sample = mean.new_empty(mean.size()).normal_()
        return mean + sample


class VUnet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        final_act = retrieve(config, "model_paras/final_act", default=False)
        nf_max = retrieve(config, "model_paras/nf_max", default=128)
        nf_start = retrieve(config, "model_paras/nf_start", default=64)
        spatial_size = retrieve(config, "model_paras/spatial_size", default=256)
        dropout_prob = retrieve(config, "model_paras/dropout_prob", default=0.1)
        img_channels = retrieve(config, "model_paras/img_channels", default=3)
        motion_channels = retrieve(config, "model_paras/motion_channels", default=2)
        clip_hist = retrieve(config, "model_paras/clip_hist", default=4)
        clip_pred = retrieve(config, "model_paras/clip_pred", default=1)
        num_flows = retrieve(config, "model_paras/num_flows", default=4)
        device = retrieve(config, "device", default="cuda:0")
        # new: drop_path_rate maximum for encoders/bottleneck
        drop_path_rate = retrieve(config, "model_paras/drop_path_rate", default=0.0)

        output_channels = img_channels * clip_pred
        n_stages = 1 + int(np.round(np.log2(spatial_size))) - 2
        conv_layer_type = Conv2d if final_act else NormConv2d
        self.f_phi = VUnetEncoder(
            n_stages=n_stages,
            nf_in=img_channels * clip_hist + motion_channels * num_flows,
            nf_start=nf_start,
            nf_max=nf_max,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
        )
        self.e_theta = VUnetEncoder(
            n_stages=n_stages,
            nf_in=motion_channels * num_flows,
            nf_start=nf_start,
            nf_max=nf_max,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
        )
        self.zc = ZConverter(
            n_stages=n_stages,
            nf=nf_max,
            device=device,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
            drop_path_rate=drop_path_rate,
        )
        self.bottleneck = VUnetBottleneck(
            n_stages=n_stages,
            nf=nf_max,
            device=device,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
            drop_path_rate=drop_path_rate,
        )
        self.decoder = VUnetDecoder(
            n_stages=n_stages,
            nf=nf_max,
            nf_out=output_channels,
            conv_layer=conv_layer_type,
            spatial_size=spatial_size,
            final_act=final_act,
            dropout_prob=dropout_prob,
        )
        self.saved_tensors = None

        # recommended: initialize weights for Linear / Conv layers
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs: dict, mode: str = "train") -> Tensor:
        x_f_in = torch.cat((inputs['appearance'], inputs['motion']), dim=1)
        x_f = self.f_phi(x_f_in)
        q_means, zs = self.zc(x_f)
        x_e = self.e_theta(inputs['motion'])
        if mode == "train":
            out_b, p_means, ps = self.bottleneck(x_e, zs)
        else:
            out_b, p_means, ps = self.bottleneck(x_e, q_means)
        out_img = self.decoder(out_b, x_f)
        self.saved_tensors = dict(q_means=q_means, p_means=p_means)
        return out_img

