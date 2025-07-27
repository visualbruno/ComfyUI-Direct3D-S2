from typing import *
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ...modules import sparse as sp
from .base import SparseTransformerBase


class SparseSubdivideBlock3d(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.act_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, padding=1),
            sp.SparseSiLU()
        )
        
        self.sub = sp.SparseSubdivide()
        
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(self.out_channels, self.out_channels, 3, padding=1),
            sp.SparseSiLU(),
        )
        
    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.act_layers(x)
        h = self.sub(h)
        h = self.out_layers(h)
        return h
    
    def forward(self, x: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseSDFDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
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
        representation_config: dict = None,
        out_channels: int = 1,
        chunk_size: int = 1,
    ):
        super().__init__(
            in_channels=latent_channels,
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
        self.resolution = resolution
        self.rep_config = representation_config
        self.out_channels = out_channels
        self.chunk_size = chunk_size
        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3d(
                channels=model_channels,
                out_channels=model_channels // 4,
                use_checkpoint=use_checkpoint,
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 4,
                out_channels=model_channels // 8,
                use_checkpoint=use_checkpoint,
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 8,
                out_channels=model_channels // 16,
                use_checkpoint=use_checkpoint,
            )
        ])

        self.out_layer = sp.SparseLinear(model_channels // 16, self.out_channels)
        self.out_active = sp.SparseTanh()

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        super().convert_to_fp16()
        self.upsample.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        super().convert_to_fp32()
        self.upsample.apply(convert_module_to_f32)  
    
    @torch.no_grad()
    def split_for_meshing(self, x: sp.SparseTensor, chunk_size=4, padding=4):
        
        sub_resolution = self.resolution // chunk_size
        upsample_ratio = 8 # hard-coded here
        assert sub_resolution % padding == 0
        out = []
        
        for i in range(chunk_size):
            for j in range(chunk_size):
                for k in range(chunk_size):
                    # Calculate padded boundaries
                    start_x = max(0, i * sub_resolution - padding)
                    end_x = min((i + 1) * sub_resolution + padding, self.resolution)
                    start_y = max(0, j * sub_resolution - padding)
                    end_y = min((j + 1) * sub_resolution + padding, self.resolution)
                    start_z = max(0, k * sub_resolution - padding)
                    end_z = min((k + 1) * sub_resolution + padding, self.resolution)
                    
                    # Store original (unpadded) boundaries for later cropping
                    orig_start_x = i * sub_resolution
                    orig_end_x = (i + 1) * sub_resolution
                    orig_start_y = j * sub_resolution
                    orig_end_y = (j + 1) * sub_resolution
                    orig_start_z = k * sub_resolution
                    orig_end_z = (k + 1) * sub_resolution

                    mask = torch.logical_and(
                        torch.logical_and(
                            torch.logical_and(x.coords[:, 1] >= start_x, x.coords[:, 1] < end_x),
                            torch.logical_and(x.coords[:, 2] >= start_y, x.coords[:, 2] < end_y)
                        ),
                        torch.logical_and(x.coords[:, 3] >= start_z, x.coords[:, 3] < end_z)
                    )

                    if mask.sum() > 0:
                        # Get the coordinates and shift them to local space
                        coords = x.coords[mask].clone()
                        # Shift to local coordinates
                        coords[:, 1:] = coords[:, 1:] - torch.tensor([start_x, start_y, start_z], 
                                                                    device=coords.device).view(1, 3)

                        chunk_tensor = sp.SparseTensor(x.feats[mask], coords)
                        # Store the boundaries and offsets as metadata for later reconstruction
                        chunk_tensor.bounds = {
                            'original': (orig_start_x * upsample_ratio, orig_end_x * upsample_ratio + (upsample_ratio - 1), orig_start_y * upsample_ratio, orig_end_y * upsample_ratio + (upsample_ratio - 1), orig_start_z * upsample_ratio, orig_end_z * upsample_ratio + (upsample_ratio - 1)),
                            'offsets': (start_x * upsample_ratio, start_y * upsample_ratio, start_z * upsample_ratio)  # Store offsets for reconstruction
                        }
                        out.append(chunk_tensor)

                    del mask
                    torch.cuda.empty_cache()
        return out
    
    @torch.no_grad()
    def split_single_chunk(self, x: sp.SparseTensor, chunk_size=4, padding=4):
        sub_resolution = self.resolution // chunk_size
        upsample_ratio = 8 # hard-coded here
        assert sub_resolution % padding == 0

        mask_sum = -1
        while mask_sum < 1:
            orig_start_x = random.randint(0, self.resolution - sub_resolution)
            orig_end_x = orig_start_x + sub_resolution
            orig_start_y = random.randint(0, self.resolution - sub_resolution)
            orig_end_y = orig_start_y + sub_resolution
            orig_start_z = random.randint(0, self.resolution - sub_resolution)
            orig_end_z = orig_start_z + sub_resolution
            start_x = max(0, orig_start_x - padding)
            end_x = min(orig_end_x + padding, self.resolution)
            start_y = max(0, orig_start_y - padding)
            end_y = min(orig_end_y + padding, self.resolution)
            start_z = max(0, orig_start_z - padding)
            end_z = min(orig_end_z + padding, self.resolution)

            mask_ori = torch.logical_and(
                torch.logical_and(
                    torch.logical_and(x.coords[:, 1] >= orig_start_x, x.coords[:, 1] < orig_end_x),
                    torch.logical_and(x.coords[:, 2] >= orig_start_y, x.coords[:, 2] < orig_end_y)
                ),
                torch.logical_and(x.coords[:, 3] >= orig_start_z, x.coords[:, 3] < orig_end_z)
            )
            mask_sum = mask_ori.sum()

        # Store the boundaries and offsets as metadata for later reconstruction
        bounds = {
            'original': (orig_start_x * upsample_ratio, orig_end_x * upsample_ratio + (upsample_ratio - 1), orig_start_y * upsample_ratio, orig_end_y * upsample_ratio + (upsample_ratio - 1), orig_start_z * upsample_ratio, orig_end_z * upsample_ratio + (upsample_ratio - 1)),
            'start': (start_x, end_x, start_y, end_y, start_z, end_z),
            'offsets': (start_x * upsample_ratio, start_y * upsample_ratio, start_z * upsample_ratio)  # Store offsets for reconstruction
        }
        return bounds
    
    def forward_single_chunk(self, x: sp.SparseTensor, padding=4):
        
        bounds = self.split_single_chunk(x, self.chunk_size, padding=padding)

        start_x, end_x, start_y, end_y, start_z, end_z = bounds['start']
        mask = torch.logical_and(
            torch.logical_and(
                torch.logical_and(x.coords[:, 1] >= start_x, x.coords[:, 1] < end_x),
                torch.logical_and(x.coords[:, 2] >= start_y, x.coords[:, 2] < end_y)
            ),
            torch.logical_and(x.coords[:, 3] >= start_z, x.coords[:, 3] < end_z)
        )

        # Shift to local coordinates
        coords = x.coords.clone()
        coords[:, 1:] = coords[:, 1:] - torch.tensor([start_x, start_y, start_z], 
                                                    device=coords.device).view(1, 3)

        chunk = sp.SparseTensor(x.feats[mask], coords[mask])
    
        chunk_result = self.upsamples(chunk)

        coords = chunk_result.coords.clone()

        # Restore global coordinates
        offsets = torch.tensor(bounds['offsets'], 
                                device=coords.device).view(1, 3)
        coords[:, 1:] = coords[:, 1:] + offsets

        # Filter points within original bounds
        original = bounds['original']
        within_bounds = torch.logical_and(
            torch.logical_and(
                torch.logical_and(
                    coords[:, 1] >= original[0],
                    coords[:, 1] < original[1]
                ),
                torch.logical_and(
                    coords[:, 2] >= original[2],
                    coords[:, 2] < original[3]
                )
            ),
            torch.logical_and(
                coords[:, 3] >= original[4],
                coords[:, 3] < original[5]
            )
        )

        final_coords = coords[within_bounds]
        final_feats = chunk_result.feats[within_bounds]

        return sp.SparseTensor(final_feats, final_coords)

    def upsamples(self, x, return_feat: bool = False):
        dtype = x.dtype
        for block in self.upsample:
            x = block(x)
        x = x.type(dtype)

        output = self.out_active(self.out_layer(x))

        if return_feat:
            return output, x
        else:
            return output
    
    def forward(self, x: sp.SparseTensor, factor: float = None, return_feat: bool = False):
        h = super().forward(x, factor)
        if self.chunk_size <= 1:
            for block in self.upsample:
                h = block(h)
            h = h.type(x.dtype)

            if return_feat:
                return self.out_active(self.out_layer(h)), h
        
            h = self.out_layer(h)
            h = self.out_active(h)
            return h
        else:
            if self.training:
                return self.forward_single_chunk(h)
            else:
                batch_size = x.shape[0]                
                chunks = self.split_for_meshing(h, chunk_size=self.chunk_size)
                all_coords, all_feats = [], []
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_result = self.upsamples(chunk)

                    for b in range(batch_size):
                        mask = torch.nonzero(chunk_result.coords[:, 0] == b).squeeze(-1)
                        if mask.numel() > 0:
                            coords = chunk_result.coords[mask].clone()

                            # Restore global coordinates
                            offsets = torch.tensor(chunk.bounds['offsets'], 
                                                    device=coords.device).view(1, 3)
                            coords[:, 1:] = coords[:, 1:] + offsets

                            # Filter points within original bounds
                            bounds = chunk.bounds['original']
                            within_bounds = torch.logical_and(
                                torch.logical_and(
                                    torch.logical_and(
                                        coords[:, 1] >= bounds[0],
                                        coords[:, 1] < bounds[1]
                                    ),
                                    torch.logical_and(
                                        coords[:, 2] >= bounds[2],
                                        coords[:, 2] < bounds[3]
                                    )
                                ),
                                torch.logical_and(
                                    coords[:, 3] >= bounds[4],
                                    coords[:, 3] < bounds[5]
                                )
                            )
                            
                            if within_bounds.any():
                                all_coords.append(coords[within_bounds])
                                all_feats.append(chunk_result.feats[mask][within_bounds])
                    
                    if not self.training:
                        torch.cuda.empty_cache()

                final_coords = torch.cat(all_coords)
                final_feats = torch.cat(all_feats)
                
                return sp.SparseTensor(final_feats, final_coords)
            