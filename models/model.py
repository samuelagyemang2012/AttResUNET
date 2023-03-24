import cv2
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from modules.modules import DoubleConv, ResBlockIN, ResBlockIN2, ConvTrans, ConvTrans2, AttentionBlock, \
    AttentionBlockIN, AttentionBlockIN2, RRCNNBlock, UpConv, Respath
import torchvision.transforms as transforms
import numpy as np


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
    def __init__(self, in_channels=3, out_channels=3):
        super(AttResUNET, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.c1 = ResBlockIN(in_channels=in_channels, out_channels=64)
        self.c2 = ResBlockIN(in_channels=64, out_channels=128)
        self.c3 = ResBlockIN(in_channels=128, out_channels=256)
        self.c4 = ResBlockIN(in_channels=256, out_channels=512)
        self.c5 = ResBlockIN(in_channels=512, out_channels=1024)

        self.u5 = ConvTrans(in_channels=1024, out_channels=512)
        self.a5 = AttentionBlockIN(F_g=512, F_l=512, F_int=256)
        self.u5_c5 = ResBlockIN(in_channels=1024, out_channels=512)

        self.u4 = ConvTrans(in_channels=512, out_channels=256)
        self.a4 = AttentionBlockIN(F_g=256, F_l=256, F_int=128)
        self.u4_c4 = ResBlockIN(in_channels=512, out_channels=256)

        self.u3 = ConvTrans(in_channels=256, out_channels=128)
        self.a3 = AttentionBlockIN(F_g=128, F_l=128, F_int=64)
        self.u3_c3 = ResBlockIN(in_channels=256, out_channels=128)

        self.u2 = ConvTrans(in_channels=128, out_channels=64)
        self.a2 = AttentionBlockIN(F_g=64, F_l=64, F_int=32)
        self.u2_c2 = ResBlockIN(in_channels=128, out_channels=64)

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


class AttResUNET2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(AttResUNET2, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.c1 = ResBlockIN2(in_channels=in_channels, out_channels=64)
        self.c2 = ResBlockIN2(in_channels=64, out_channels=128)
        self.c3 = ResBlockIN2(in_channels=128, out_channels=256)
        self.c4 = ResBlockIN2(in_channels=256, out_channels=512)
        self.c5 = ResBlockIN2(in_channels=512, out_channels=1024)

        self.u5 = ConvTrans2(in_channels=1024, out_channels=512)
        self.a5 = AttentionBlockIN2(F_g=512, F_l=512, F_int=256)
        self.u5_c5 = ResBlockIN2(in_channels=1024, out_channels=512)

        self.u4 = ConvTrans2(in_channels=512, out_channels=256)
        self.a4 = AttentionBlockIN2(F_g=256, F_l=256, F_int=128)
        self.u4_c4 = ResBlockIN2(in_channels=512, out_channels=256)

        self.u3 = ConvTrans2(in_channels=256, out_channels=128)
        self.a3 = AttentionBlockIN2(F_g=128, F_l=128, F_int=64)
        self.u3_c3 = ResBlockIN2(in_channels=256, out_channels=128)

        self.u2 = ConvTrans2(in_channels=128, out_channels=64)
        self.a2 = AttentionBlockIN2(F_g=64, F_l=64, F_int=32)
        self.u2_c2 = ResBlockIN2(in_channels=128, out_channels=64)

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


class R2AttUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, t=2):
        super(R2AttUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNNBlock(in_channels=in_channels, out_channels=64, t=t)

        self.RRCNN2 = RRCNNBlock(in_channels=64, out_channels=128, t=t)

        self.RRCNN3 = RRCNNBlock(in_channels=128, out_channels=256, t=t)

        self.RRCNN4 = RRCNNBlock(in_channels=256, out_channels=512, t=t)

        self.RRCNN5 = RRCNNBlock(in_channels=512, out_channels=1024, t=t)

        self.Up5 = UpConv(in_channels=1024, out_channels=512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNNBlock(in_channels=1024, out_channels=512, t=t)

        self.Up4 = UpConv(in_channels=512, out_channels=256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNNBlock(in_channels=512, out_channels=256, t=t)

        self.Up3 = UpConv(in_channels=256, out_channels=128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNNBlock(in_channels=256, out_channels=128, t=t)

        self.Up2 = UpConv(in_channels=128, out_channels=64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNNBlock(in_channels=128, out_channels=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class XNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(XNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # self.up = ConvTrans()

        self.conv1 = ResBlockIN(in_channels, 64)  # DoubleConv(in_channels, 64)
        self.conv2 = ResBlockIN(64, 128)  # DoubleConv(64, 128)
        self.conv3 = ResBlockIN(128, 256)  # DoubleConv(128, 256)

        self.conv4 = ResBlockIN(256, 512)  # DoubleConv(256, 512)
        self.conv5 = ResBlockIN(512, 512)  # DoubleConv(512, 512)

        self.conv6 = ResBlockIN(512, 256)  # DoubleConv(512, 256)
        self.conv7 = ResBlockIN(256, 128)  # DoubleConv(256, 128)

        self.conv8 = ResBlockIN(128, 128)  # DoubleConv(128, 128)
        self.conv9 = ResBlockIN(128, 256)  # DoubleConv(128, 256)
        self.conv10 = ResBlockIN(256, 512)  # DoubleConv(256, 512)
        self.conv11 = ResBlockIN(512, 512)  # DoubleConv(512, 512)
        self.conv12 = ResBlockIN(512, 256)  # DoubleConv(512, 256)
        self.conv13 = ResBlockIN(256, 128)  # DoubleConv(256, 128)
        self.conv14 = ResBlockIN(128, 64)  # DoubleConv(128, 64)

        self.conv15 = nn.Conv2d(64, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.a1 = AttentionBlockIN(256, 256, 128)
        self.a2 = AttentionBlockIN(128, 128, 64)
        self.a3 = AttentionBlockIN(64, 64, 64)

        self.cat_conv1 = ResBlockIN(512, 256)  # DoubleConv(512, 256)
        self.cat_conv2 = ResBlockIN(256, 128)  # DoubleConv(256, 128)
        self.cat_conv3 = ResBlockIN(128, 64)  # DoubleConv(128, 64)

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


class XNet2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(XNet2, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # self.up = ConvTrans()

        self.conv1 = ResBlockIN(in_channels, 64)  # DoubleConv(in_channels, 64)
        self.conv2 = ResBlockIN(64, 128)  # DoubleConv(64, 128)
        self.conv3 = ResBlockIN(128, 256)  # DoubleConv(128, 256)

        self.conv4 = ResBlockIN(256, 512)  # DoubleConv(256, 512)
        self.conv5 = ResBlockIN(512, 512)  # DoubleConv(512, 512)

        self.conv6 = ResBlockIN(512, 256)  # DoubleConv(512, 256)
        self.conv7 = ResBlockIN(256, 128)  # DoubleConv(256, 128)

        self.conv8 = ResBlockIN(128, 128)  # DoubleConv(128, 128)
        self.conv9 = ResBlockIN(128, 256)  # DoubleConv(128, 256)
        self.conv10 = ResBlockIN(256, 512)  # DoubleConv(256, 512)
        self.conv11 = ResBlockIN(512, 512)  # DoubleConv(512, 512)
        self.conv12 = ResBlockIN(512, 256)  # DoubleConv(512, 256)
        self.conv13 = ResBlockIN(256, 128)  # DoubleConv(256, 128)
        self.conv14 = ResBlockIN(128, 64)  # DoubleConv(128, 64)

        self.conv15 = nn.Conv2d(64, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.a1 = AttentionBlockIN(256, 256, 128)
        self.a2 = AttentionBlockIN(128, 128, 64)
        self.a3 = AttentionBlockIN(64, 64, 64)

        self.cat_conv1 = ResBlockIN(512, 256)  # DoubleConv(512, 256)
        self.cat_conv2 = ResBlockIN(256, 128)  # DoubleConv(256, 128)
        self.cat_conv3 = ResBlockIN(128, 64)  # DoubleConv(128, 64)

        self.respath1 = Respath(256, 256)
        self.respath2 = Respath(128, 128)
        self.respath3 = Respath(64, 64)

    def forward(self, x):
        c1 = self.conv1(x)  # [1, 64, 400, 400]
        mr_c1 = self.respath3(c1)  # [1,64,400,400]
        c1_pool = self.pool(c1)  # [1, 64, 200, 200]

        c2 = self.conv2(c1_pool)  # [1, 128, 200, 200]
        mr_c2 = self.respath2(c2)  # [1, 128, 200, 200]
        c2_pool = self.pool(c2)  # [1, 128, 100, 100]

        c3 = self.conv3(c2_pool)  # [1, 256, 100, 100]
        mr_c3 = self.respath1(c3)  # [1, 256, 100, 100]
        c3_pool = self.pool(c3)  # [1, 256, 50, 50]

        c4 = self.conv4(c3_pool)  # [1, 512, 50, 50]
        c5 = self.conv5(c4)  # [1, 512, 50, 50]

        up6 = self.up(c5)  # [1, 512, 100, 100]
        c6 = self.conv6(up6)  # [1, 256, 100, 100]
        a1 = self.a1(c6, mr_c3)  # [1, 256, 100, 100]
        c6 = torch.cat((c3, a1), dim=1)  # [1, 512, 100, 100]
        c6 = self.cat_conv1(c6)  # [1, 256, 100, 100]

        up7 = self.up(c6)  # [1, 256, 200, 200]
        c7 = self.conv7(up7)  # [1, 128, 200, 200]
        a2 = self.a2(c7, mr_c2)  # [1, 128, 200, 200]
        c7 = torch.cat((c2, a2), dim=1)  # [1, 256, 200, 200]
        c7 = self.cat_conv2(c7)  # [1, 128, 200, 200]

        c8 = self.conv8(c7)  # [1, 128, 200, 200]
        mr_c8 = self.respath2(c8)  # [1, 128, 200, 200]
        c8_pool = self.pool(c8)  # [1, 128, 100, 100]

        c9 = self.conv9(c8_pool)  # [1, 256, 100, 100]
        mr_c9 = self.respath1(c9)  # [1, 256,100,100]
        c9_pool = self.pool(c9)  # [1, 256, 50, 50]

        c10 = self.conv10(c9_pool)  # [1, 512, 50, 50]
        c11 = self.conv11(c10)  # [1, 512, 50, 50]

        up12 = self.up(c11)  # [1, 512, 100, 100]
        c12 = self.conv12(up12)  # [1, 256, 100, 100]
        a3 = self.a1(c12, mr_c9)  # [1, 256, 100, 100]
        c12 = torch.cat((c9, a3), dim=1)  # [1, 512, 100, 100]
        c12 = self.cat_conv1(c12)  # [1, 256, 100, 100]

        up13 = self.up(c12)  # [1, 256, 200, 200]
        c13 = self.conv13(up13)  # [1, 128, 200, 200]
        a4 = self.a2(c13, mr_c8)  # [1, 128, 200, 200]
        c13 = torch.cat((c8, a4), dim=1)  # [1, 256, 200, 200]
        c13 = self.cat_conv2(c13)  # [1, 128, 200, 200]

        up14 = self.up(c13)  # [1, 128, 400, 400]
        c14 = self.conv14(up14)  # [1, 64, 400, 400]
        a4 = self.a3(c14, mr_c1)  # [1, 64, 400, 400]
        c14 = torch.cat((c1, a4), dim=1)  # [1, 128, 400, 400]
        c14 = self.cat_conv3(c14)  # [1, 64, 400, 400]

        c15 = self.conv15(c14)  # [1, 3, 400, 400]

        return c15


def test():
    x = torch.randn((1, 3, 400, 400))
    model = XNet2(in_channels=3, out_channels=3)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
