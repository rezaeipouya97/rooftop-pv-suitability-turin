"""
model.py - TorchSparse 2.0.0b UNet for 3-class point cloud segmentation.

CRITICAL FIX: TorchSparse 2.0.0b transposed convolutions need kernel maps
(kmaps) that live on the SparseTensor from the encoder's strided convs.
Creating a NEW SparseTensor destroys these maps.

Solution: in skip connections, we modify upsampled.F IN-PLACE (concatenate
skip features onto the existing tensor). This keeps coords, stride, and
kmaps intact.

Input:  10-dim (xyz_rel + rgb + normals + HAG)
Output: per-voxel logits, num_classes channels
"""

import torch
import torch.nn as nn

import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        spnn.Conv3d(in_ch, out_ch, kernel_size=3),
        spnn.BatchNorm(out_ch),
        spnn.ReLU(True),
        spnn.Conv3d(out_ch, out_ch, kernel_size=3),
        spnn.BatchNorm(out_ch),
        spnn.ReLU(True),
    )


def _down_block(ch):
    return nn.Sequential(
        spnn.Conv3d(ch, ch, kernel_size=2, stride=2),
        spnn.BatchNorm(ch),
        spnn.ReLU(True),
    )


class TorchSparseUNet(nn.Module):
    """4-level encoder-decoder with skip connections. 32->64->128->256."""

    def __init__(self, in_channels=10, num_classes=3):
        super().__init__()
        ch = [32, 64, 128, 256]

        # Encoder
        self.enc1  = _conv_block(in_channels, ch[0])
        self.down1 = _down_block(ch[0])
        self.enc2  = _conv_block(ch[0], ch[1])
        self.down2 = _down_block(ch[1])
        self.enc3  = _conv_block(ch[1], ch[2])
        self.down3 = _down_block(ch[2])

        # Bottleneck
        self.bottleneck = _conv_block(ch[2], ch[3])

        # Decoder
        self.up3  = spnn.Conv3d(ch[3], ch[2], kernel_size=2, stride=2, transposed=True)
        self.dec3 = _conv_block(ch[2] + ch[2], ch[2])

        self.up2  = spnn.Conv3d(ch[2], ch[1], kernel_size=2, stride=2, transposed=True)
        self.dec2 = _conv_block(ch[1] + ch[1], ch[1])

        self.up1  = spnn.Conv3d(ch[1], ch[0], kernel_size=2, stride=2, transposed=True)
        self.dec1 = _conv_block(ch[0] + ch[0], ch[0])

        self.final = spnn.Conv3d(ch[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        bn = self.bottleneck(self.down3(e3))

        # Decoder - NEVER create new SparseTensors.
        # Modify .F in-place on the upsampled tensor to preserve kmaps.
        d3 = self.up3(bn)
        d3.F = torch.cat([d3.F, e3.F], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2.F = torch.cat([d2.F, e2.F], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1.F = torch.cat([d1.F, e1.F], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    import numpy as np

    print(f"TorchSparse version: {torchsparse.__version__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TorchSparseUNet(in_channels=10, num_classes=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    rng = np.random.default_rng(42)
    N = 5000
    coords = rng.integers(-50, 50, size=(N, 3)).astype(np.int32)
    coords_unique, idx = np.unique(coords, axis=0, return_index=True)
    feats = rng.standard_normal((len(idx), 10)).astype(np.float32)

    # TorchSparse 2.0 expects N x 4 coords: [batch, x, y, z]
    # TorchSparse 2.0.0b: coords are [x, y, z, batch] — batch LAST
    batch_col = np.zeros((len(idx), 1), dtype=np.int32)
    coords_4d = np.hstack([coords_unique, batch_col])

    x = SparseTensor(
        coords=torch.from_numpy(coords_4d).int().to(device),
        feats=torch.from_numpy(feats).float().to(device),
    )

    with torch.no_grad():
        out = model(x)

    print(f"  Input:  {x.F.shape[0]} voxels, {x.F.shape[1]} channels")
    print(f"  Output: {out.F.shape[0]} voxels, {out.F.shape[1]} channels")
    print(f"  Range:  [{out.F.min().item():.3f}, {out.F.max().item():.3f}]")
    print("Self-test PASSED")


if __name__ == "__main__":
    self_test()
