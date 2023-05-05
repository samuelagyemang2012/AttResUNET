import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import math


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResBlockM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, padding_mode='zeros', dilation=1,
                 dropout=0.2):
        super(ResBlockM, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dropout = dropout
        self.padding_mode = padding_mode

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, self.kernel_size, self.stride, self.padding, bias=False,
                      dilation=self.dilation, padding_mode=self.padding_mode),
            nn.BatchNorm2d(out_channels)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, self.kernel_size, self.stride, self.padding, bias=False,
                      dilation=self.dilation, padding_mode=self.padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, self.kernel_size, self.stride, self.padding, bias=False,
                      dilation=self.dilation, padding_mode=self.padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        x = self.conv(x) + self.skip(x)
        x = nn.ReLU(inplace=True)(x)

        return x


class UpConvM(nn.Module):
    def __init__(self, in_channels, out_channels, mode='bilinear', use_batchnorm=True):
        super(UpConvM, self).__init__()

        self.use_batchnorm = use_batchnorm

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.use_batchnorm:
            x = self.up1(x)
        else:
            x = self.up2(x)

        return x


class UpConvX(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super(UpConvX, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)  # in_c * 4, H, W --> in_c, H*2, W*2
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class ConvTransM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(ConvTransM, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)

        return x


class AttentionBlockM(nn.Module):
    def __init__(self, F_g, F_l, F_int, use_batchnorm=True):
        super(AttentionBlockM, self).__init__()

        self.use_batchnorm = use_batchnorm

        self.W_g1 = nn.Sequential(  # 512,256
            nn.Conv2d(F_g, F_int, 1, 1, 0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x1 = nn.Sequential(  # 512, 256
            nn.Conv2d(F_l, F_int, 1, 1, 0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi1 = nn.Sequential(  # 256
            nn.Conv2d(F_int, 1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        ##################################################
        self.W_g2 = nn.Sequential(  # 512,256
            nn.Conv2d(F_g, F_int, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(F_int)
        )

        self.W_x2 = nn.Sequential(  # 512, 256
            nn.Conv2d(F_l, F_int, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(F_int)
        )

        self.psi2 = nn.Sequential(  # 256
            nn.Conv2d(F_int, 1, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if self.use_batchnorm:
            g1 = self.W_g1(g)  # 512
            x1 = self.W_x1(x)  # 256
            psi = self.relu(g1 + x1)
            psi = self.psi1(psi)
        else:
            g1 = self.W_g2(g)  # 512
            x1 = self.W_x2(x)  # 256
            psi = self.relu(g1 + x1)
            psi = self.psi2(psi)

        return x * psi


class GhostModuleM(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, stride=1, ratio=2, dw_size=3, dropout=0.2, use_batchnorm=True):
        super(GhostModuleM, self).__init__()
        self.oup = oup
        self.use_batchnorm = use_batchnorm

        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv1 = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )

        self.cheap_operation1 = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.primary_conv2 = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.InstanceNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )

        self.cheap_operation2 = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.InstanceNorm2d(new_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        if self.use_batchnorm:
            x1 = self.primary_conv1(x)
            x2 = self.cheap_operation1(x1)
            out = torch.cat([x1, x2], dim=1)
        else:
            x1 = self.primary_conv2(x)
            x2 = self.cheap_operation2(x1)
            out = torch.cat([x1, x2], dim=1)

        return out[:, :self.oup, :, :]


def test():
    x = torch.randn((1, 64, 256, 256))
    net = UpConvX(in_channels=64)
    preds = net(x)

    print(preds.shape)


if __name__ == "__main__":
    test()
