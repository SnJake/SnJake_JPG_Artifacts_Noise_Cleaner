import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch, out_ch, stride=1, groups=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, groups=groups, bias=True)


class ResidualBlock(nn.Module):
    def __init__(self, ch, groups=1, expansion=2):
        super().__init__()
        mid = ch * expansion
        self.conv1 = conv3x3(ch, mid, groups=groups)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = conv3x3(mid, ch, groups=groups)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out + identity


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = F.pixel_shuffle(x, 2)
        return self.act(x)


class CBAM(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(ch // r, 8), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(ch // r, 8), ch, 1, bias=True),
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x))
        x = x * ca
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))
        return x * sa


class UNetBlock(nn.Module):
    def __init__(self, ch, n_blocks=2, use_cbam=False):
        super().__init__()
        layers = []
        for _ in range(n_blocks):
            layers.append(ResidualBlock(ch))
        self.body = nn.Sequential(*layers)
        self.cbam = CBAM(ch) if use_cbam else None

    def forward(self, x):
        x = self.body(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x


class UNetRestorer(nn.Module):
    """
    Lightweight UNet with residual blocks and optional CBAM attention.
    Designed for 16GB VRAM with reasonable batch sizes and 256px crops.
    """

    def __init__(
        self,
        in_ch=3,
        base_ch=64,
        depths=(2, 2, 4, 4),
        use_cbam=(False, True, True, False),
        out_ch=3,
    ):
        super().__init__()
        self.entry = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        # Encoder
        self.enc0 = UNetBlock(chs[0], n_blocks=depths[0], use_cbam=use_cbam[0])
        self.down0 = Down(chs[0], chs[1])
        self.enc1 = UNetBlock(chs[1], n_blocks=depths[1], use_cbam=use_cbam[1])
        self.down1 = Down(chs[1], chs[2])
        self.enc2 = UNetBlock(chs[2], n_blocks=depths[2], use_cbam=use_cbam[2])
        self.down2 = Down(chs[2], chs[3])
        self.enc3 = UNetBlock(chs[3], n_blocks=depths[3], use_cbam=use_cbam[3])

        # Decoder
        self.up2 = Up(chs[3], chs[2])
        self.dec2 = UNetBlock(chs[2] + chs[2], n_blocks=2, use_cbam=False)
        self.up1 = Up(chs[2] + chs[2], chs[1])
        self.dec1 = UNetBlock(chs[1] + chs[1], n_blocks=2, use_cbam=False)
        self.up0 = Up(chs[1] + chs[1], chs[0])
        self.dec0 = UNetBlock(chs[0] + chs[0], n_blocks=2, use_cbam=False)

        self.exit = nn.Conv2d(chs[0] + chs[0], out_ch, 3, padding=1)

    def forward(self, x):
        x0 = self.entry(x)
        e0 = self.enc0(x0)
        x1 = self.down0(e0)
        e1 = self.enc1(x1)
        x2 = self.down1(e1)
        e2 = self.enc2(x2)
        x3 = self.down2(e2)
        e3 = self.enc3(x3)

        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        d0 = torch.cat([d0, e0], dim=1)
        d0 = self.dec0(d0)

        out = self.exit(d0)
        # Residual learning: predict clean minus input, then add back input
        return x + out
