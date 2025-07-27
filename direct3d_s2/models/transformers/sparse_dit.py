from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ...modules.transformer import AbsolutePositionEmbedder
from ...modules import sparse as sp
from ...modules.sparse.transformer.modulated import ModulatedSparseTransformerCrossBlock
from .dense_dit import TimestepEmbedder
    

class SparseDiT(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        num_kv_heads: Optional[int] = 2,
        compression_block_size: int = 4,
        selection_block_size: int = 8,
        topk: int = 8,
        compression_version: str = 'v2',
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        sparse_conditions: bool = False,
        factor: float = 1.0,
        window_size: Optional[int] = 8,
        use_shift: bool = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.sparse_conditions = sparse_conditions
        self.factor = factor
        self.compression_block_size = compression_block_size
        self.selection_block_size = selection_block_size

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )
        
        if sparse_conditions:
            self.cond_proj = sp.SparseLinear(cond_channels, cond_channels)
            self.pos_embedder_cond = AbsolutePositionEmbedder(model_channels, in_channels=3)

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)

        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                num_kv_heads=num_kv_heads,
                compression_block_size=compression_block_size,
                selection_block_size=selection_block_size,
                topk=topk,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                compression_version=compression_version,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                resolution=resolution,
                window_size=window_size,
                shift_window=window_size // 2 * (_ % 2) if use_shift else window_size // 2,
            )
            for _ in range(num_blocks)
        ])
        
        self.out_layer = sp.SparseLinear(model_channels, out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        # self.blocks.apply(convert_module_to_f16)
        self.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: Union[torch.Tensor, sp.SparseTensor]) -> sp.SparseTensor:
        h = self.input_layer(x).type(self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        if self.sparse_conditions:
            cond = self.cond_proj(cond)
            cond = cond + self.pos_embedder_cond(cond.coords[:, 1:]).type(self.dtype)
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:], factor=self.factor).type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(x.dtype))
        return h
