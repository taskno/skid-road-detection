import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def crop_or_pad(self, x, target_size):
        _, _, h, w = x.size()
        th, tw = target_size
        dh, dw = th - h, tw - w

        pad_top = max(dh // 2, 0)
        pad_bottom = max(dh - pad_top, 0)
        pad_left = max(dw // 2, 0)
        pad_right = max(dw - pad_left, 0)

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        if dh < 0 or dw < 0:
            x = x[:, :, pad_top:h+dh, pad_left:w+dw]

        return x

    def forward(self, x):
        x1 = self.enc1(x)       # 64
        x2 = self.enc2(self.pool(x1))  # 128
        x3 = self.enc3(self.pool(x2))  # 256
        x4 = self.enc4(self.pool(x3))  # 512

        x5 = self.bottleneck(self.pool(x4))  # 1024

        x = self.up4(x5)
        x4 = self.crop_or_pad(x4, x.shape[2:])
        x = self.dec4(torch.cat([x, x4], dim=1))

        x = self.up3(x)
        x3 = self.crop_or_pad(x3, x.shape[2:])
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x2 = self.crop_or_pad(x2, x.shape[2:])
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x1 = self.crop_or_pad(x1, x.shape[2:])
        x = self.dec1(torch.cat([x, x1], dim=1))

        #return torch.sigmoid(self.final(x))
        x = self.final(x)
        x = F.interpolate(x, size=(150, 150), mode='bilinear', align_corners=False)
        return torch.sigmoid(x)