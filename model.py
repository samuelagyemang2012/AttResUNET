import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from modules.modules import DoubleConv, ResBlockIN, ConvTrans, AttentionBlock, \
    AttentionBlockIN


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
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

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

        self.c1x1 = nn.Conv2d(64, out_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.c1(x)

        x2 = self.pool(x1)
        x2 = self.c2(x2)

        x3 = self.pool(x2)
        x3 = self.c3(x3)

        x4 = self.pool(x3)
        x4 = self.c4(x4)

        x5 = self.pool(x4)
        x5 = self.c5(x5)

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


def test():
    x = torch.randn((1, 3, 400, 400))
    model = AttResUNET(in_channels=3, out_channels=3)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
