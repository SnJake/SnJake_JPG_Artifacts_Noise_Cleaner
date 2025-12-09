import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """
    Channel-first LayerNorm for tensors shaped (N, C, H, W).
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, bias=True),
        )

        self.sg = SimpleGate()

        self.conv4 = nn.Conv2d(c, c * ffn_expand, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(c * ffn_expand // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class UNetRestorer(nn.Module):
    """
    NAFNet-based UNet architecture (v2).
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 32, enc_num=None, middle_num: int = 12, dec_num=None):
        super().__init__()
        if enc_num is None:
            enc_num = [2, 2, 4, 8]
        if dec_num is None:
            dec_num = [2, 2, 2, 2]

        self.intro = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = base_ch

        for num in enc_num:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(Downsample(chan, chan * 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_num)])

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        for num in dec_num:
            self.ups.append(Upsample(chan, chan // 2))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.ending = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.intro(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        out = self.ending(x)
        return out + inp


__all__ = ["UNetRestorer", "NAFBlock", "LayerNorm2d"]
