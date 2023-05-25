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


#####################################################################################################################

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            **kwargs,
            bias=True,
        )
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv2d(in_c, in_c, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))


class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        for i in range(5):
            self.blocks.append(
                ConvBlock(
                    in_channels + channels * i,
                    channels if i <= 3 else in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_act=True if i <= 3 else False,
                )
            )

    def forward(self, x):
        new_inputs = x
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)
        return self.residual_beta * out + x


class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels) for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x


######################################################################################################################
class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=True)
        self.act = nn.ReLU() if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))


class ConvBlock3(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=True)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, **kwargs, bias=True)
        self.cbam = CBAM(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.cnn(x)
        x = self.cbam(x)
        x = self.cnn2(x)
        return self.act(x)


class UpsampleBlock2(nn.Module):
    def __init__(self, in_c, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv2d(in_c, in_c, 3, 1, 1, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))


class DenseResidualBlock2(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        for i in range(5):
            self.blocks.append(
                ConvBlock2(
                    in_channels=in_channels + channels * i,
                    out_channels=channels if i <= 3 else in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_act=True if i <= 3 else False,
                )
            )

    def forward(self, x):
        new_inputs = x
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)
        return self.residual_beta * out + x


class DenseResidualBlock3(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        for i in range(5):
            self.blocks.append(
                ConvBlock3(
                    in_channels=in_channels + channels * i,
                    out_channels=channels if i <= 3 else in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

    def forward(self, x):
        new_inputs = x
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)
        return self.residual_beta * out + x


class RRDB2(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseResidualBlock2(in_channels) for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x


class RRDB3(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseResidualBlock3(in_channels) for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x


######################################################################################################################
class ChannelAttention(nn.Module):
    def __init__(self, ch=32, ratio=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(ch, ch // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // ratio, ch, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x1 = self.mlp(x1)

        x2 = self.max_pool(x).squeeze(-1).squeeze(-1)
        x2 = self.mlp(x2)

        feats = x1 + x2
        feats = self.sigmoid(feats).unsqueeze(-1).unsqueeze(-1)
        refined_feats = x * feats

        return refined_feats


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)

        feats = torch.cat([x1, x2], dim=1)
        feats = self.conv(feats)
        feats = self.sigmoid(feats)
        refined_feats = x * feats

        return refined_feats


class ca_stem_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        )

        # CA
        self.channel_att = ChannelAttention(ch=64, ratio=16)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Shortcut
        a = self.conv(x)
        a = self.channel_att(a)

        b = self.shortcut(x)
        print(b.shape)
        return a + b


def test():
    x = torch.randn((1, 64, 100, 100))
    # net = GhostModuleM(inp=64, oup=256, kernel_size=3, ratio=2, dw_size=3, stride=1)
    net = CBAM(in_channels=64)
    preds = net(x)

    print(preds.shape)


if __name__ == "__main__":
    test()
