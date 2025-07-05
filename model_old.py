import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_msssim import ssim 
import torchvision.transforms.functional as TF


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

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
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.att = AttentionGate(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        if x1.shape[2:] != x2.shape[2:]:
            x1 = TF.resize(x1, size=x2.shape[2:], antialias=True)

        x2_att = self.att(g=x1, x=x2)
        x = torch.cat([x2_att, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class AttentionResUNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_features=64):
        super(AttentionResUNet, self).__init__()
        self.inc = ConvBlock(in_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)
        
        self.up1 = Up(base_features * 8, base_features * 4)
        self.up2 = Up(base_features * 4, base_features * 2)
        self.up3 = Up(base_features * 2, base_features)
        self.outc = OutConv(base_features, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.transform = nn.functional.interpolate

    def forward(self, pred, target):
        pred_vgg = self.vgg(pred.repeat(1, 3, 1, 1))
        with torch.no_grad():
            target_vgg = self.vgg(target.repeat(1, 3, 1, 1))
        return self.criterion(pred_vgg, target_vgg)


class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        pred_float32 = pred.to(torch.float32)
        target_float32 = target.to(torch.float32)

        pred_fft = torch.fft.rfft2(pred_float32, norm='backward')
        target_fft = torch.fft.rfft2(target_float32, norm='backward')
        return self.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class GrandUnifiedLoss(nn.Module):
    def __init__(self, device, lambda_l1=0.15, lambda_ssim=0.85, lambda_fft=0.1, lambda_perc=0.05):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim 
        self.lambda_fft = lambda_fft
        self.lambda_perc = lambda_perc
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(device)
        self.fft_loss = FFTLoss()

    def forward(self, pred, target):
        loss_l1 = self.l1_loss(pred, target)
        loss_ssim = 1 - ssim(pred, target, data_range=1.0, size_average=True)
        loss_perc = self.perceptual_loss(pred, target)
        loss_fft = self.fft_loss(pred, target)
        
        total_loss = (self.lambda_l1 * loss_l1 +
                      self.lambda_ssim * loss_ssim +
                      self.lambda_perc * loss_perc +
                      self.lambda_fft * loss_fft)
        
        return total_loss, loss_l1, loss_ssim, loss_perc, loss_fft
