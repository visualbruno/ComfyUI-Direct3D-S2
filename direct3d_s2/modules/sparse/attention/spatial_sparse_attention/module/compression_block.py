import torch.nn as nn
import direct3d_s2.modules.sparse as sp


class SparseDownBlock3d_v1(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: int = None,
        factor: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sp.SparseConv3d(self.out_channels, self.out_channels, 1, padding=0),
            sp.SparseSiLU()
        )
        self.down = sp.SparseDownsample(factor)
        
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.act_layers(x)
        h = self.down(h)
        return h

class SparseDownBlock3d_v2(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: int = None,
        num_groups: int = 32,
        factor: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels),
            sp.SparseSiLU()
        )

        self.down = sp.SparseDownsample(factor)
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
        
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.act_layers(x)
        h = self.down(h)
        x = self.down(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        return h