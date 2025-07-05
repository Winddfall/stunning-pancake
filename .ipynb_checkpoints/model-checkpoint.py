import torch
import torch.nn as nn
from pytorch_msssim import SSIM # Ensure SSIM class is imported, not the ssim function


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # This addition will now work because g1 and x1 will have the same dimensions
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv_skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.inc = ResidualBlock(1, 64)
        self.pool = nn.MaxPool2d(2)
        self.down1 = ResidualBlock(64, 128)
        self.down2 = ResidualBlock(128, 256)
        self.down3 = ResidualBlock(256, 512)

        # Decoder
        # ❗️ FIX #1: Removed 'output_padding=1' to ensure spatial dimensions match the skip connection.
        self.up0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att0 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.conv0 = ResidualBlock(512, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.conv1 = ResidualBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.conv2 = ResidualBlock(128, 64)

        # ❗️ FIX #2: Changed input channels from 32 to 64 to match the output of self.conv2.
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))

        # Decoder
        d0 = self.up0(x4)
        x3_att = self.att0(g=d0, x=x3)
        d0 = torch.cat([x3_att, d0], dim=1)
        d0 = self.conv0(d0)

        d1 = self.up1(d0)
        x2_att = self.att1(g=d1, x=x2)
        d1 = torch.cat([x2_att, d1], dim=1)
        d1 = self.conv1(d1)

        d2 = self.up2(d1)
        x1_att = self.att2(g=d2, x=x1)
        d2 = torch.cat([x1_att, d2], dim=1)
        d2 = self.conv2(d2)

        return self.sigmoid(self.outc(d2))


class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        pred_fft_abs = torch.abs(pred_fft)
        target_fft_abs = torch.abs(target_fft)
        return self.l1_loss(pred_fft_abs, target_fft_abs)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.05, beta=0.9, gamma=0.05): # L1, SSIM, FFT
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)
        self.fft_loss = FFTLoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = 1 - self.ssim_loss(pred, target)
        fft = self.fft_loss(pred, target)
        return self.alpha * l1 + self.beta * ssim + self.gamma * fft