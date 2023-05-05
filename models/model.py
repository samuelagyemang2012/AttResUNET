import cv2
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from modules.modules import DoubleConv, AttentionBlockM, ResBlockM, ConvTransM, GhostModuleM, UpConvM, UpConvX
import torchvision.transforms as transforms
import numpy as np
from utils import load_checkpoint


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=(2, 2), stride=(2, 2)))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=(1, 1))

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class AttResUNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, use_batchnorm=False):
        super(AttResUNET, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.c1 = ResBlockM(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.c2 = ResBlockM(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.c3 = ResBlockM(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.c4 = ResBlockM(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.c5 = ResBlockM(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)

        self.u5 = ConvTransM(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=1)
        self.a5 = AttentionBlockM(F_g=512, F_l=512, F_int=256, use_batchnorm=use_batchnorm)
        self.u5_c5 = ResBlockM(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.u4 = ConvTransM(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=1)
        self.a4 = AttentionBlockM(F_g=256, F_l=256, F_int=128, use_batchnorm=use_batchnorm)
        self.u4_c4 = ResBlockM(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.u3 = ConvTransM(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=1)
        self.a3 = AttentionBlockM(F_g=128, F_l=128, F_int=64, use_batchnorm=use_batchnorm)
        self.u3_c3 = ResBlockM(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.u2 = ConvTransM(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=1)
        self.a2 = AttentionBlockM(F_g=64, F_l=64, F_int=32, use_batchnorm=use_batchnorm)
        self.u2_c2 = ResBlockM(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.c1x1 = nn.Conv2d(64, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        # downsample
        x1 = self.c1(x)

        x2 = self.pool(x1)
        x2 = self.c2(x2)

        x3 = self.pool(x2)
        x3 = self.c3(x3)

        x4 = self.pool(x3)
        x4 = self.c4(x4)

        x5 = self.pool(x4)
        x5 = self.c5(x5)

        # upsample
        d5 = self.u5(x5)
        x4 = self.a5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.u5_c5(d5)

        d4 = self.u4(d5)
        x3 = self.a4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.u4_c4(d4)

        d3 = self.u3(d4)
        x2 = self.a3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.u3_c3(d3)

        d2 = self.u2(d3)
        x1 = self.a2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.u2_c2(d2)

        d1 = self.c1x1(d2)

        return d1


class XNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(XNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = ResBlockM(in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = ResBlockM(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = ResBlockM(128, 256, kernel_size=3, stride=1, padding=1)

        self.conv4 = ResBlockM(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = ResBlockM(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv6 = ResBlockM(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = ResBlockM(256, 128, kernel_size=3, stride=1, padding=1)

        self.conv8 = ResBlockM(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv9 = ResBlockM(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv10 = ResBlockM(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv11 = ResBlockM(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = ResBlockM(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv13 = ResBlockM(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv14 = ResBlockM(128, 64, kernel_size=3, stride=1, padding=1)

        self.conv15 = nn.Conv2d(64, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.a1 = AttentionBlockM(256, 256, 128)
        self.a2 = AttentionBlockM(128, 128, 64)
        self.a3 = AttentionBlockM(64, 64, 32)

        self.cat_conv1 = ResBlockM(512, 256, kernel_size=3, stride=1, padding=1)  # DoubleConv(512, 256)
        self.cat_conv2 = ResBlockM(256, 128, kernel_size=3, stride=1, padding=1)  # DoubleConv(256, 128)
        self.cat_conv3 = ResBlockM(128, 64, kernel_size=3, stride=1, padding=1)  # DoubleConv(128, 64)

    def forward(self, x):
        c1 = self.conv1(x)  # [1, 64, 400, 400]
        c1_pool = self.pool(c1)  # [1, 64, 200, 200]

        c2 = self.conv2(c1_pool)  # [1, 128, 200, 200]
        c2_pool = self.pool(c2)  # [1, 128, 100, 100]

        c3 = self.conv3(c2_pool)  # [1, 256, 100, 100]
        c3_pool = self.pool(c3)  # [1, 256, 50, 50]

        c4 = self.conv4(c3_pool)  # [1, 512, 50, 50]
        c5 = self.conv5(c4)  # [1, 512, 50, 50]

        up6 = self.up(c5)  # [1, 512, 100, 100]
        c6 = self.conv6(up6)  # [1, 256, 100, 100]
        a1 = self.a1(c6, c3)  # [1, 256, 100, 100]
        c6 = torch.cat((c3, a1), dim=1)  # [1, 512, 100, 100]
        c6 = self.cat_conv1(c6)  # [1, 256, 100, 100]

        up7 = self.up(c6)  # [1, 256, 200, 200]
        c7 = self.conv7(up7)  # [1, 128, 200, 200]
        a2 = self.a2(c7, c2)  # [1, 128, 200, 200]
        c7 = torch.cat((c2, a2), dim=1)  # [1, 256, 200, 200]
        c7 = self.cat_conv2(c7)  # [1, 128, 200, 200]

        c8 = self.conv8(c7)  # [1, 128, 200, 200]
        c8_pool = self.pool(c8)  # [1, 128, 100, 100]

        c9 = self.conv9(c8_pool)  # [1, 256, 100, 100]
        c9_pool = self.pool(c9)  # [1, 256, 50, 50]

        c10 = self.conv10(c9_pool)  # [1, 512, 50, 50]
        c11 = self.conv11(c10)  # [1, 512, 50, 50]

        up12 = self.up(c11)  # [1, 512, 100, 100]
        c12 = self.conv12(up12)  # [1, 256, 100, 100]
        a3 = self.a1(c12, c9)  # [1, 256, 100, 100]
        c12 = torch.cat((c9, a3), dim=1)  # [1, 512, 100, 100]
        c12 = self.cat_conv1(c12)  # [1, 256, 100, 100]

        up13 = self.up(c12)  # [1, 256, 200, 200]
        c13 = self.conv13(up13)  # [1, 128, 200, 200]
        a4 = self.a2(c13, c8)  # [1, 128, 200, 200]
        c13 = torch.concat((c8, a4), dim=1)  # [1, 256, 200, 200]
        c13 = self.cat_conv2(c13)  # [1, 128, 200, 200]

        up14 = self.up(c13)  # [1, 128, 400, 400]
        c14 = self.conv14(up14)  # [1, 64, 400, 400]
        a4 = self.a3(c14, c1)  # [1, 64, 400, 400]
        c14 = torch.cat((c1, a4), dim=1)  # [1, 128, 400, 400]
        c14 = self.cat_conv3(c14)  # [1, 64, 400, 400]

        c15 = self.conv15(c14)  # [1, 3, 400, 400]

        return c15


class Network5(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dropout=0.2, use_batch=False):
        """
        Initializes each part of the convolutional neural network.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Downsampling
        self.conv1 = ResBlockM(in_channels, 32, kernel_size=3, stride=1, padding=1, dropout=dropout,
                               padding_mode='reflect')
        self.conv2 = ResBlockM(32, 64, kernel_size=3, stride=1, padding=1, dropout=dropout)
        self.conv3 = ResBlockM(64, 128, kernel_size=3, stride=1, padding=1, dropout=dropout)
        self.conv4 = ResBlockM(128, 256, kernel_size=3, stride=1, padding=1, dropout=dropout)
        self.conv5 = ResBlockM(256, 512, kernel_size=3, stride=1, padding=1, dropout=dropout)

        # Upsampling
        self.up1 = UpConvM(512, 256, use_batchnorm=use_batch)
        self.up_c1 = ResBlockM(512, 256, kernel_size=3, stride=1, padding=1, dropout=dropout)

        self.up2 = UpConvM(256, 128, use_batchnorm=use_batch)
        self.up_c2 = ResBlockM(256, 128, kernel_size=3, stride=1, padding=1, dropout=dropout)

        self.up3 = UpConvM(128, 64, use_batchnorm=use_batch)
        self.up_c3 = ResBlockM(128, 64, kernel_size=3, stride=1, padding=1, dropout=dropout)

        self.up4 = UpConvM(64, 32, use_batchnorm=use_batch)
        self.up_c4 = ResBlockM(64, 32, kernel_size=3, stride=1, padding=1, dropout=dropout)

        self.head = nn.Conv2d(32, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

        # self.output = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)  # [1, 32, 400, 400]
        x1_pool = self.pool(x1)  # [1, 32, 200, 200]

        x2 = self.conv2(x1_pool)  # [1, 64, 200, 200]
        x2_pool = self.pool(x2)  # [1, 64, 100, 100]

        x3 = self.conv3(x2_pool)  # [1, 128, 100, 100]
        x3_pool = self.pool(x3)  # [1, 128, 50, 50]

        x4 = self.conv4(x3_pool)  # [1, 256, 50, 50]
        x4_pool = self.pool(x4)  # [1, 256, 25, 25]

        # bottle_neck
        x5 = self.conv5(x4_pool)  # [1, 512, 25, 25]

        # bottle_neck
        # x6 = self.conv6(x5)  # [1, 512, 25, 25]

        x7 = self.up1(x5)  # [1, 256, 50, 50]
        x7 = torch.cat((x4, x7), dim=1)  # [1, 512, 50, 50]
        x7 = self.up_c1(x7)  # [1, 256, 50, 50]

        x8 = self.up2(x7)  # [1, 128, 100, 100]
        x8 = torch.cat((x3, x8), dim=1)  # [1, 256, 100, 100]
        x8 = self.up_c2(x8)  # [1, 128, 100, 100]

        x9 = self.up3(x8)  # [1, 64, 200, 200]]
        x9 = torch.cat((x2, x9), dim=1)  # [1, 128, 200, 200]
        x9 = self.up_c3(x9)  # [1, 64, 200, 200]

        x10 = self.up4(x9)  # [1, 32, 400, 400]
        x10 = torch.cat((x1, x10), dim=1)  # [1, 64, 400, 400])
        x10 = self.up_c4(x10)  # [1, 32, 400, 400]

        head = self.head(x10)  # [1, 3, 400, 400]

        return head


class Network7(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dropout=0.2, use_batchnorm=False):
        """
        Initializes each part of the convolutional neural network.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Downsampling
        self.conv1 = GhostModuleM(in_channels, 32, kernel_size=3, stride=1, use_batchnorm=use_batchnorm)
        self.conv2 = GhostModuleM(32, 64, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv3 = GhostModuleM(64, 128, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv4 = GhostModuleM(128, 256, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv5 = GhostModuleM(256, 512, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        # Bottleneck
        # self.conv6 = ResBlockM(512, 512, kernel_size=4, stride=1, padding=3, dilation=2, with_affine=affine,
        #                        use_batch=use_batch)

        # Upsampling

        self.up1 = UpConvM(512, 256, use_batchnorm=use_batchnorm)
        self.up_c1 = GhostModuleM(512, 256, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up2 = UpConvM(256, 128, use_batchnorm=use_batchnorm)
        self.up_c2 = GhostModuleM(256, 128, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up3 = UpConvM(128, 64, use_batchnorm=use_batchnorm)
        self.up_c3 = GhostModuleM(128, 64, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up4 = UpConvM(64, 32, use_batchnorm=use_batchnorm)
        self.up_c4 = GhostModuleM(64, 32, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.head = nn.Conv2d(32, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        x1 = self.conv1(x)  # [1, 32, 400, 400]
        x1_pool = self.pool(x1)  # [1, 32, 200, 200]

        x2 = self.conv2(x1_pool)  # [1, 64, 200, 200]
        x2_pool = self.pool(x2)  # [1, 64, 100, 100]

        x3 = self.conv3(x2_pool)  # [1, 128, 100, 100]
        x3_pool = self.pool(x3)  # [1, 128, 50, 50]

        x4 = self.conv4(x3_pool)  # [1, 256, 50, 50]
        x4_pool = self.pool(x4)  # [1, 256, 25, 25]

        # bottle_neck
        x5 = self.conv5(x4_pool)  # [1, 512, 25, 25]

        x7 = self.up1(x5)  # [1, 256, 50, 50]
        x7 = torch.cat((x4, x7), dim=1)  # [1, 512, 50, 50]
        x7 = self.up_c1(x7)  # [1, 256, 50, 50])

        x8 = self.up2(x7)  # [1, 128, 100, 100]
        x8 = torch.cat((x3, x8), dim=1)  # [1, 256, 100, 100]
        x8 = self.up_c2(x8)  # [1, 128, 100, 100]

        x9 = self.up3(x8)  # [1, 64, 200, 200]]
        x9 = torch.cat((x2, x9), dim=1)  # [1, 128, 200, 200]
        x9 = self.up_c3(x9)  # [1, 64, 200, 200]

        x10 = self.up4(x9)  # [1, 32, 400, 400]
        x10 = torch.cat((x1, x10), dim=1)  # [1, 64, 400, 400])
        x10 = self.up_c4(x10)  # [1, 32, 400, 400]

        pred = self.head(x10)  # [1, 3, 400, 400]

        return pred


class Network7V2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dropout=0.2, use_batchnorm=False):
        """
        Initializes each part of the convolutional neural network.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Downsampling
        self.conv1 = GhostModuleM(in_channels, 32, kernel_size=3, stride=1, use_batchnorm=use_batchnorm)
        self.conv2 = GhostModuleM(32, 64, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv3 = GhostModuleM(64, 128, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv4 = GhostModuleM(128, 256, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv5 = GhostModuleM(256, 512, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        # Bottleneck
        # self.conv6 = ResBlockM(512, 512, kernel_size=4, stride=1, padding=3, dilation=2, with_affine=affine,
        #                        use_batch=use_batch)

        # Upsampling

        self.up1 = UpConvM(512, 256, use_batchnorm=True)
        self.up_c1 = ResBlockM(512, 256, kernel_size=3, stride=1, dropout=dropout, padding=1)

        self.up2 = UpConvM(256, 128, use_batchnorm=True)
        self.up_c2 = ResBlockM(256, 128, kernel_size=3, stride=1, dropout=dropout, padding=1)

        self.up3 = UpConvM(128, 64, use_batchnorm=True)
        self.up_c3 = ResBlockM(128, 64, kernel_size=3, stride=1, dropout=dropout, padding=1)

        self.up4 = UpConvM(64, 32, use_batchnorm=True)
        self.up_c4 = ResBlockM(64, 32, kernel_size=3, stride=1, dropout=dropout, padding=1)

        self.head = nn.Conv2d(32, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        x1 = self.conv1(x)  # [1, 32, 400, 400]
        x1_pool = self.pool(x1)  # [1, 32, 200, 200]

        x2 = self.conv2(x1_pool)  # [1, 64, 200, 200]
        x2_pool = self.pool(x2)  # [1, 64, 100, 100]

        x3 = self.conv3(x2_pool)  # [1, 128, 100, 100]
        x3_pool = self.pool(x3)  # [1, 128, 50, 50]

        x4 = self.conv4(x3_pool)  # [1, 256, 50, 50]
        x4_pool = self.pool(x4)  # [1, 256, 25, 25]

        # bottle_neck
        x5 = self.conv5(x4_pool)  # [1, 512, 25, 25]

        x7 = self.up1(x5)  # [1, 256, 50, 50]
        x7 = torch.cat((x4, x7), dim=1)  # [1, 512, 50, 50]
        x7 = self.up_c1(x7)  # [1, 256, 50, 50])

        x8 = self.up2(x7)  # [1, 128, 100, 100]
        x8 = torch.cat((x3, x8), dim=1)  # [1, 256, 100, 100]
        x8 = self.up_c2(x8)  # [1, 128, 100, 100]

        x9 = self.up3(x8)  # [1, 64, 200, 200]]
        x9 = torch.cat((x2, x9), dim=1)  # [1, 128, 200, 200]
        x9 = self.up_c3(x9)  # [1, 64, 200, 200]

        x10 = self.up4(x9)  # [1, 32, 400, 400]
        x10 = torch.cat((x1, x10), dim=1)  # [1, 64, 400, 400])
        x10 = self.up_c4(x10)  # [1, 32, 400, 400]

        pred = self.head(x10)  # [1, 3, 400, 400]

        return pred


class Network7DeBlur(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dropout=0.2, use_batchnorm=False):
        """
        Initializes each part of the convolutional neural network.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Downsampling
        self.conv1 = GhostModuleM(in_channels, 32, kernel_size=3, stride=1, use_batchnorm=use_batchnorm)
        self.conv2 = GhostModuleM(32, 64, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv3 = GhostModuleM(64, 128, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv4 = GhostModuleM(128, 256, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv5 = GhostModuleM(256, 512, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        # Bottleneck
        # self.conv6 = ResBlockM(512, 512, kernel_size=4, stride=1, padding=3, dilation=2, with_affine=affine,
        #                        use_batch=use_batch)

        # Upsampling

        self.up1 = UpConvM(512, 256, use_batchnorm=use_batchnorm)
        self.up_c1 = GhostModuleM(512, 256, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up2 = UpConvM(256, 128, use_batchnorm=use_batchnorm)
        self.up_c2 = GhostModuleM(256, 128, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up3 = UpConvM(128, 64, use_batchnorm=use_batchnorm)
        self.up_c3 = GhostModuleM(128, 64, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up4 = UpConvM(64, 32, use_batchnorm=use_batchnorm)
        self.up_c4 = GhostModuleM(64, 32, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.pred = nn.Conv2d(32, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.deblur = DeBlur()

    def forward(self, x):
        x1 = self.conv1(x)  # [1, 32, 400, 400]
        x1_pool = self.pool(x1)  # [1, 32, 200, 200]

        x2 = self.conv2(x1_pool)  # [1, 64, 200, 200]
        x2_pool = self.pool(x2)  # [1, 64, 100, 100]

        x3 = self.conv3(x2_pool)  # [1, 128, 100, 100]
        x3_pool = self.pool(x3)  # [1, 128, 50, 50]

        x4 = self.conv4(x3_pool)  # [1, 256, 50, 50]
        x4_pool = self.pool(x4)  # [1, 256, 25, 25]

        # bottle_neck
        x5 = self.conv5(x4_pool)  # [1, 512, 25, 25]

        x7 = self.up1(x5)  # [1, 256, 50, 50]
        x7 = torch.cat((x4, x7), dim=1)  # [1, 512, 50, 50]
        x7 = self.up_c1(x7)  # [1, 256, 50, 50])

        x8 = self.up2(x7)  # [1, 128, 100, 100]
        x8 = torch.cat((x3, x8), dim=1)  # [1, 256, 100, 100]
        x8 = self.up_c2(x8)  # [1, 128, 100, 100]

        x9 = self.up3(x8)  # [1, 64, 200, 200]]
        x9 = torch.cat((x2, x9), dim=1)  # [1, 128, 200, 200]
        x9 = self.up_c3(x9)  # [1, 64, 200, 200]

        x10 = self.up4(x9)  # [1, 32, 400, 400]
        x10 = torch.cat((x1, x10), dim=1)  # [1, 64, 400, 400])
        x10 = self.up_c4(x10)  # [1, 32, 400, 400]

        pred = self.pred(x10)  # [1, 3, 400, 400]

        final = self.deblur(pred)

        return final


class Network7DeBlur2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dropout=0.2, use_batchnorm=False, deblur_weights=None):
        """
        Initializes each part of the convolutional neural network.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Downsampling
        self.conv1 = GhostModuleM(in_channels, 32, kernel_size=3, stride=1, use_batchnorm=use_batchnorm)
        self.conv2 = GhostModuleM(32, 64, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv3 = GhostModuleM(64, 128, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv4 = GhostModuleM(128, 256, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv5 = GhostModuleM(256, 512, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        # Bottleneck
        # self.conv6 = ResBlockM(512, 512, kernel_size=4, stride=1, padding=3, dilation=2, with_affine=affine,
        #                        use_batch=use_batch)

        # Upsampling

        self.up1 = UpConvM(512, 256, use_batchnorm=use_batchnorm)
        self.up_c1 = GhostModuleM(512, 256, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up2 = UpConvM(256, 128, use_batchnorm=use_batchnorm)
        self.up_c2 = GhostModuleM(256, 128, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up3 = UpConvM(128, 64, use_batchnorm=use_batchnorm)
        self.up_c3 = GhostModuleM(128, 64, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up4 = UpConvM(64, 32, use_batchnorm=use_batchnorm)
        self.up_c4 = GhostModuleM(64, 32, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.pred = nn.Conv2d(32, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

        if deblur_weights is None:
            self.deblur = DeBlur(use_batch=True)
        else:
            print("loading deblur weights")
            self.deblur = DeBlur(use_batch=True)
            weights = torch.load(deblur_weights)
            self.deblur = load_checkpoint(weights, self.deblur)
            # net = net.to(cfg.DEVICE)

    def forward(self, x):
        x1 = self.conv1(x)  # [1, 32, 400, 400]
        x1_pool = self.pool(x1)  # [1, 32, 200, 200]

        x2 = self.conv2(x1_pool)  # [1, 64, 200, 200]
        x2_pool = self.pool(x2)  # [1, 64, 100, 100]

        x3 = self.conv3(x2_pool)  # [1, 128, 100, 100]
        x3_pool = self.pool(x3)  # [1, 128, 50, 50]

        x4 = self.conv4(x3_pool)  # [1, 256, 50, 50]
        x4_pool = self.pool(x4)  # [1, 256, 25, 25]

        # bottle_neck
        x5 = self.conv5(x4_pool)  # [1, 512, 25, 25]

        x7 = self.up1(x5)  # [1, 256, 50, 50]
        x7 = torch.cat((x4, x7), dim=1)  # [1, 512, 50, 50]
        x7 = self.up_c1(x7)  # [1, 256, 50, 50])

        x8 = self.up2(x7)  # [1, 128, 100, 100]
        x8 = torch.cat((x3, x8), dim=1)  # [1, 256, 100, 100]
        x8 = self.up_c2(x8)  # [1, 128, 100, 100]

        x9 = self.up3(x8)  # [1, 64, 200, 200]]
        x9 = torch.cat((x2, x9), dim=1)  # [1, 128, 200, 200]
        x9 = self.up_c3(x9)  # [1, 64, 200, 200]

        x10 = self.up4(x9)  # [1, 32, 400, 400]
        x10 = torch.cat((x1, x10), dim=1)  # [1, 64, 400, 400])
        x10 = self.up_c4(x10)  # [1, 32, 400, 400]

        pred = self.pred(x10)  # [1, 3, 400, 400]

        final = self.deblur(pred)

        return final


class Network7Att(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dropout=0.2, use_batchnorm=True):
        """
        Initializes each part of the convolutional neural network.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Downsampling
        self.conv1 = GhostModuleM(in_channels, 32, kernel_size=3, stride=1, use_batchnorm=use_batchnorm)
        self.conv2 = GhostModuleM(32, 64, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv3 = GhostModuleM(64, 128, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv4 = GhostModuleM(128, 256, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv5 = GhostModuleM(256, 512, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        # Bottleneck
        # self.conv6 = ResBlockM(512, 512, kernel_size=4, stride=1, padding=3, dilation=2, with_affine=affine,
        #                        use_batch=use_batch)

        # Upsampling

        self.up1 = UpConvM(512, 256, use_batchnorm=use_batchnorm)
        self.att1 = AttentionBlockM(F_g=256, F_l=256, F_int=128, use_batchnorm=use_batchnorm)
        self.up_c1 = GhostModuleM(512, 256, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up2 = UpConvM(256, 128, use_batchnorm=use_batchnorm)
        self.att2 = AttentionBlockM(F_g=128, F_l=128, F_int=64)
        self.up_c2 = GhostModuleM(256, 128, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up3 = UpConvM(128, 64, use_batchnorm=use_batchnorm)
        self.att3 = AttentionBlockM(F_g=64, F_l=64, F_int=32, use_batchnorm=use_batchnorm)
        self.up_c3 = GhostModuleM(128, 64, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up4 = UpConvM(64, 32, use_batchnorm=use_batchnorm)
        self.att4 = AttentionBlockM(F_g=32, F_l=32, F_int=16, use_batchnorm=use_batchnorm)
        self.up_c4 = GhostModuleM(64, 32, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.head = nn.Conv2d(32, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        x1 = self.conv1(x)  # [1, 32, 400, 400]
        x1_pool = self.pool(x1)  # [1, 32, 200, 200]

        x2 = self.conv2(x1_pool)  # [1, 64, 200, 200]
        x2_pool = self.pool(x2)  # [1, 64, 100, 100]

        x3 = self.conv3(x2_pool)  # [1, 128, 100, 100]
        x3_pool = self.pool(x3)  # [1, 128, 50, 50]

        x4 = self.conv4(x3_pool)  # [1, 256, 50, 50]
        x4_pool = self.pool(x4)  # [1, 256, 25, 25]

        # bottle_neck
        x5 = self.conv5(x4_pool)  # [1, 512, 25, 25]

        x7 = self.up1(x5)  # [1, 256, 50, 50]
        x7 = self.att1(g=x7, x=x4)  # [1, 256, 50, 50]
        x7 = torch.cat((x4, x7), dim=1)  # [1, 512, 50, 50]
        x7 = self.up_c1(x7)  # [1, 256, 50, 50])

        x8 = self.up2(x7)  # [1, 128, 100, 100]
        x8 = self.att2(g=x8, x=x3)  # [1, 128, 100, 100]
        x8 = torch.cat((x3, x8), dim=1)  # [1, 256, 100, 100]
        x8 = self.up_c2(x8)  # [1, 128, 100, 100]

        x9 = self.up3(x8)  # [1, 64, 200, 200]
        x9 = self.att3(g=x9, x=x2)  # [1, 64, 200, 200]
        x9 = torch.cat((x2, x9), dim=1)  # [1, 128, 200, 200]
        x9 = self.up_c3(x9)  # [1, 64, 200, 200]

        x10 = self.up4(x9)  # [1, 32, 400, 400]
        x10 = self.att4(g=x10, x=x1)  # [1, 32, 400, 400]
        x10 = torch.cat((x1, x10), dim=1)  # [1, 64, 400, 400])
        x10 = self.up_c4(x10)  # [1, 32, 400, 400]

        head = self.head(x10)  # [1, 3, 400, 400]

        return head


class Network7L(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dropout=0.2, use_batchnorm=False):
        """
        Initializes each part of the convolutional neural network.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Downsampling
        self.conv1 = GhostModuleM(in_channels, 64, kernel_size=3, stride=1, use_batchnorm=use_batchnorm)
        self.conv2 = GhostModuleM(64, 128, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv3 = GhostModuleM(128, 256, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv4 = GhostModuleM(256, 512, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv5 = GhostModuleM(512, 1024, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        # Bottleneck
        # self.conv6 = ResBlockM(512, 512, kernel_size=4, stride=1, padding=3, dilation=2, with_affine=affine,
        #                        use_batch=use_batch)

        # Upsampling

        self.up5 = UpConvM(1024, 512, use_batchnorm=use_batchnorm)
        self.up_c5 = GhostModuleM(1024, 512, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up4 = UpConvM(512, 256, use_batchnorm=use_batchnorm)
        self.up_c4 = GhostModuleM(512, 256, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up3 = UpConvM(256, 128, use_batchnorm=use_batchnorm)
        self.up_c3 = GhostModuleM(256, 128, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.up2 = UpConvM(128, 64, use_batchnorm=use_batchnorm)
        self.up_c2 = GhostModuleM(128, 64, kernel_size=3, stride=1, dropout=dropout, use_batchnorm=use_batchnorm)

        self.head = nn.Conv2d(64, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        x1 = self.conv1(x)  # [1, 64, 400, 400]
        x1_pool = self.pool(x1)  # [1, 64, 200, 200]

        x2 = self.conv2(x1_pool)  # [1, 128, 200, 200]
        x2_pool = self.pool(x2)  # [1, 128, 100, 100]

        x3 = self.conv3(x2_pool)  # [1, 256, 100, 100]
        x3_pool = self.pool(x3)  # [1, 256, 50, 50]

        x4 = self.conv4(x3_pool)  # [1, 512, 50, 50]
        x4_pool = self.pool(x4)  # [1, 512, 25, 25]

        # bottle_neck
        x5 = self.conv5(x4_pool)  # [1, 1024, 25, 25]

        x7 = self.up5(x5)  # [1, 512, 50, 50]
        x7 = torch.cat((x4, x7), dim=1)  # [1, 512, 50, 50]
        x7 = self.up_c5(x7)  # [1, 256, 50, 50])

        x8 = self.up4(x7)  # [1, 128, 100, 100]
        x8 = torch.cat((x3, x8), dim=1)  # [1, 256, 100, 100]
        x8 = self.up_c4(x8)  # [1, 128, 100, 100]

        x9 = self.up3(x8)  # [1, 64, 200, 200]]
        x9 = torch.cat((x2, x9), dim=1)  # [1, 128, 200, 200]
        x9 = self.up_c3(x9)  # [1, 64, 200, 200]

        x10 = self.up2(x9)  # [1, 32, 400, 400]
        x10 = torch.cat((x1, x10), dim=1)  # [1, 64, 400, 400])
        x10 = self.up_c2(x10)  # [1, 32, 400, 400]

        head = self.head(x10)  # [1, 3, 400, 400]

        return head


class DeBlur(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dropout=0.2, use_batch=False):
        super(DeBlur, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = ResBlockM(in_channels, 32, kernel_size=3, stride=1, padding=1, dropout=dropout,
                               padding_mode='reflect')
        self.conv2 = ResBlockM(32, 64, kernel_size=3, stride=1, padding=1, dropout=dropout)
        self.conv3 = ResBlockM(64, 128, kernel_size=3, stride=1, padding=1, dropout=dropout)

        # Upsampling
        self.up1 = UpConvM(128, 64, use_batchnorm=use_batch)
        self.up_c1 = ResBlockM(128, 64, kernel_size=3, stride=1, padding=1, dropout=dropout)

        self.up2 = UpConvM(64, 32, use_batchnorm=use_batch)
        self.up_c2 = ResBlockM(64, 32, kernel_size=3, stride=1, padding=1, dropout=dropout)

        self.head = nn.Conv2d(32, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        x1 = self.conv1(x)  # [1, 32, 400, 400]
        x1_pool = self.pool(x1)  # [1, 32, 200, 200]

        x2 = self.conv2(x1_pool)  # [1, 64, 200, 200]
        x2_pool = self.pool(x2)  # [1, 64, 100, 100]

        x3 = self.conv3(x2_pool)  # [1, 128, 100, 100]

        x4 = self.up1(x3)  # [1, 46, 200, 200]
        x4 = torch.cat((x2, x4), dim=1)  # [1, 128, 200, 200]
        x4 = self.up_c1(x4)  # [1, 64, 200, 200]

        x5 = self.up2(x4)  # [1, 32, 400, 400]
        x5 = torch.cat((x1, x5), dim=1)  # [1, 64, 400, 400]
        x5 = self.up_c2(x5)  # [1, 32, 400, 400]

        out = self.head(x5)

        return out


class DeBlur2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dropout=0.2, use_batch=False):
        super(DeBlur2, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = ResBlockM(in_channels, 32, kernel_size=3, stride=1, padding=1, dropout=dropout,
                               padding_mode='reflect')
        self.conv2 = ResBlockM(32, 64, kernel_size=3, stride=1, padding=1, dropout=dropout)
        self.conv3 = ResBlockM(64, 128, kernel_size=3, stride=1, padding=1, dropout=dropout)

        # Upsampling
        self.up1 = UpConvM(128, 64, use_batchnorm=use_batch)
        self.up_c1 = ResBlockM(128, 64, kernel_size=3, stride=1, padding=1, dropout=dropout)

        self.up2 = UpConvM(64, 32, use_batchnorm=use_batch)
        self.up_c2 = ResBlockM(64, 32, kernel_size=3, stride=1, padding=1, dropout=dropout)

        self.head = nn.Conv2d(32, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        x1 = self.conv1(x)  # [1, 32, 400, 400]
        x1_pool = self.pool(x1)  # [1, 32, 200, 200]

        x2 = self.conv2(x1_pool)  # [1, 64, 200, 200]
        x2_pool = self.pool(x2)  # [1, 64, 100, 100]

        x3 = self.conv3(x2_pool)  # [1, 128, 100, 100]

        x4 = self.up1(x3)  # [1, 46, 200, 200]
        x4 = torch.cat((x2, x4), dim=1)  # [1, 128, 200, 200]
        x4 = self.up_c1(x4)  # [1, 64, 200, 200]

        x5 = self.up2(x4)  # [1, 32, 400, 400]
        x5 = torch.cat((x1, x5), dim=1)  # [1, 64, 400, 400]
        x5 = self.up_c2(x5)  # [1, 32, 400, 400]

        out = self.head(x5)

        return out


def test():
    x = torch.randn((1, 3, 400, 400))
    # model = Network7(in_channels=3, out_channels=3, dropout=0.2, use_batchnorm=False)
    model = Network7V2()  # in_channels=3, out_channels=3, dropout=0.2, use_batchnorm=True)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
