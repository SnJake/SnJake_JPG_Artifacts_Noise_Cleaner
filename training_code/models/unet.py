import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """
    LayerNorm, который работает с тензорами формата (N, C, H, W).
    Обычный nn.LayerNorm ожидает (N, H, W, C).
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        self.conv4 = nn.Conv2d(c, c * FFN_Expand, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(c * FFN_Expand // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # PixelShuffle with upscale=2 reduces channels by 4, so we need out_ch*4 here
        # to end up with out_ch channels after shuffling.
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class UNetRestorer(nn.Module):
    """
    NAFNet-based UNet architecture.
    SOTA for Image Restoration (Deblurring, Denoising).
    """
    def __init__(self, in_ch=3, out_ch=3, base_ch=32, enc_num=[2, 2, 4, 8], middle_num=12, dec_num=[2, 2, 2, 2]):
        super().__init__()

        self.intro = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = base_ch
        
        # enc_num определяет количество блоков на каждом уровне
        for num in enc_num:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(Downsample(chan, chan * 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_num)])

        # Decoder
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # Обратный порядок для декодера
        for num in dec_num:
            self.ups.append(Upsample(chan, chan // 2))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.ending = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x):
        inp = x  # save for global residual
        x = self.intro(x)
        
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip # Additive skip connection (NAFNet style) or Cat? 
            # NAFNet usually uses Additive for efficiency, but UNet uses Cat. 
            # Let's use simple addition to keep channels consistent with base_ch logic 
            # and save VRAM. If you want Cat, you need to adjust convs.
            # Here: Upsample halves channels, so x is [B, C, H, W]. enc_skip is [B, C, H, W].
            x = decoder(x)

        out = self.ending(x)
        return out + inp # Global Residual
