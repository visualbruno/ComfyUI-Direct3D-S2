from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from .base import SparseTransformerBase


class SparseDownBlock3d(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels),
            sp.SparseSiLU()
        )

        self.down = sp.SparseDownsample(2)
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, padding=1),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            sp.SparseConv3d(self.out_channels, self.out_channels, 3, padding=1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(channels, self.out_channels, 1)

        self.use_checkpoint = use_checkpoint
        
    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.act_layers(x)
        h = self.down(h)
        x = self.down(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        return h

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseSDFEncoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )

        self.input_layer1 = sp.SparseLinear(1, model_channels // 16)
        
        self.downsample = nn.ModuleList([
            SparseDownBlock3d(
                channels=model_channels//16,
                out_channels=model_channels // 8,
                use_checkpoint=use_checkpoint,
            ),
            SparseDownBlock3d(
                channels=model_channels // 8,
                out_channels=model_channels // 4,
                use_checkpoint=use_checkpoint,
            ),
            SparseDownBlock3d(
                channels=model_channels // 4,
                out_channels=model_channels,
                use_checkpoint=use_checkpoint,
            )
        ])

        self.resolution = resolution
        self.out_layer = sp.SparseLinear(model_channels, 2 * latent_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor, factor: float = None):

        x = self.input_layer1(x)
        for block in self.downsample:
            x = block(x)
        h = super().forward(x, factor)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        
        return h