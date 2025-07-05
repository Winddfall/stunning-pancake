import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_msssim import ssim 

class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""
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
    """下采样: maxpool + doubleconv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class OutConv(nn.Module):
    """输出卷积层"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()  # 添加Sigmoid确保输出在0～1之间
        )

    def forward(self, x):
        return self.conv(x)

##########################################################################
## 标准U-Net - 更深层网络，可选预训练权重
##########################################################################

class StandardUNet(nn.Module):
    """标准U-Net，更深层的网络结构"""
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, use_pretrained=False):
        super(StandardUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_pretrained = use_pretrained

        # 编码器 - 标准U-Net通道数
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)      # 28x28 -> 14x14
        self.down2 = Down(128, 256)     # 14x14 -> 7x7
        self.down3 = Down(256, 512)     # 7x7 -> 3x3 (对28x28适配)

        # 瓶颈层
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # 3x3 -> 1x1

        # 解码器
        self.up1 = StandardUp(1024, 512 // factor, bilinear)  # 1x1 -> 3x3
        self.up2 = StandardUp(512, 256 // factor, bilinear)   # 3x3 -> 7x7
        self.up3 = StandardUp(256, 128 // factor, bilinear)   # 7x7 -> 14x14
        self.up4 = StandardUp(128, 64, bilinear)              # 14x14 -> 28x28

        # 多尺度特征融合
        self.multiscale_fusion = MultiscaleFusion(64)

        # 细化模块
        self.refinement = RefinementModule(64)

        # 输出层
        self.outc = OutConv(64, n_classes)

        # 初始化权重
        if use_pretrained:
            self._load_pretrained_weights()
        else:
            self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_weights(self):
        """加载预训练权重（如果有的话）"""
        # 这里可以加载ImageNet预训练的编码器权重
        # 由于MNIST是单通道，需要适配
        print("🔄 加载预训练权重...")
        # 实际使用时可以加载具体的预训练模型
        pass

    def forward(self, x):
        residual = x

        # 编码器
        x1 = self.inc(x)        # 64, 28, 28
        x2 = self.down1(x1)     # 128, 14, 14
        x3 = self.down2(x2)     # 256, 7, 7
        x4 = self.down3(x3)     # 512, 3, 3
        x5 = self.down4(x4)     # 1024, 1, 1

        # 解码器（带跳跃连接）
        x = self.up1(x5, x4)    # 512, 3, 3
        x = self.up2(x, x3)     # 256, 7, 7
        x = self.up3(x, x2)     # 128, 14, 14
        x = self.up4(x, x1)     # 64, 28, 28

        # 多尺度特征融合
        x = self.multiscale_fusion(x)

        # 细化模块
        x = self.refinement(x)

        # 输出 + 残差连接
        return self.outc(x)

class StandardUp(nn.Module):
    """标准上采样模块"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class MultiscaleFusion(nn.Module):
    """多尺度特征融合模块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, 1)
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(channels, channels, 5, padding=2)
        # self.fusion = nn.Conv2d(channels * 3, channels, 1)
        self.fusion = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        # fused = torch.cat([x1, x3, x5], dim=1)
        fused = x1 + x3 + x5
        out = self.fusion(fused)
        out = self.bn(out)
        out = self.relu(out + x)  # 残差连接
        return out

class RefinementModule(nn.Module):
    """细化模块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual  # 残差连接
        out = self.relu(out)
        return out

def create_ori_unet(device: torch.device) -> nn.Module:
    """创建标准U-Net模型"""
    model = StandardUNet(
        n_channels=1,
        n_classes=1,
        bilinear=True,           # 使用双线性插值上采样
        use_pretrained=False     # 是否使用预训练权重
    ).to(device)
    
    return model

# 损失
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
    def __init__(self, alpha=0.1, beta=0.8, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1_loss = nn.L1Loss()
        self.fft_loss = FFTLoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
        fft = self.fft_loss(pred, target)
        return self.alpha * l1 + self.beta * ssim_loss + self.gamma * fft