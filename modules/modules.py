import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


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


class DoubleConvPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvPool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x) + self.skip(x)
        x = nn.ReLU(inplace=True)(x)
        return x


class ResBlockIN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockIN, self).__init__()

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.conv(x) + self.skip(x)
        x = nn.LeakyReLU(inplace=True)(x)
        return x


class ResBlockIN2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockIN2, self).__init__()

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x) + self.skip(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UpConv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ConvTrans(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvTrans, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            # nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ConvTrans2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvTrans2, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            # nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(  # 512,256
            nn.Conv2d(F_g, F_int, 1, 1, 0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(  # 512, 256
            nn.Conv2d(F_l, F_int, 1, 1, 0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(  # 256
            nn.Conv2d(F_int, 1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)  # 512
        x1 = self.W_x(x)  # 256
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionBlockIN(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlockIN, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, 1, 0),
            nn.InstanceNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, 1, 0),
            nn.InstanceNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, 1, 0),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionBlockIN2(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlockIN2, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, 1, 0),
            nn.InstanceNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, 1, 0),
            nn.InstanceNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, 1, 0),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class RecurrentBlock(nn.Module):
    def __init__(self, out_channels, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.ch_out = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(RRCNNBlock, self).__init__()
        self.RCNN = nn.Sequential(
            RecurrentBlock(out_channels, t=t),
            RecurrentBlock(out_channels, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class Conv2d_batchnorm(torch.nn.Module):
    '''
    2D Convolutional layers
    Arguments:
        num_in_filters {int} -- number of input filters
        num_out_filters {int} -- number of output filters
        kernel_size {tuple} -- size of the convolving kernel
        stride {tuple} -- stride of the convolution (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
    '''

    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride=(1, 1), activation='relu'):
        super().__init__()
        self.activation = activation
        self.conv1 = torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size,
                                     stride=stride, padding='same')
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        if self.activation == 'relu':
            return torch.nn.functional.relu(x)
        else:
            return x


class MultiResBlock(torch.nn.Module):
    '''
    MultiRes Block

    Arguments:
        num_in_channels {int} -- Number of channels coming into mutlires block
        num_filters {int} -- Number of filters in a corrsponding UNet stage
        alpha {float} -- alpha hyperparameter (default: 1.67)

    '''

    def __init__(self, in_channels, num_filters, alpha=1.67):
        super().__init__()
        self.alpha = alpha
        self.W = num_filters * alpha

        filt_cnt_3x3 = int(self.W * 0.167)
        filt_cnt_5x5 = int(self.W * 0.333)
        filt_cnt_7x7 = int(self.W * 0.5)
        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7

        self.shortcut = Conv2d_batchnorm(in_channels, num_out_filters, kernel_size=(1, 1), activation='None')

        self.conv_3x3 = Conv2d_batchnorm(in_channels, filt_cnt_3x3, kernel_size=(3, 3), activation='relu')

        self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size=(3, 3), activation='relu')

        self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size=(3, 3), activation='relu')

        self.batch_norm1 = torch.nn.BatchNorm2d(num_out_filters)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        shrtct = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = torch.cat([a, b, c], axis=1)
        x = self.batch_norm1(x)

        x = x + shrtct
        x = self.batch_norm2(x)
        x = torch.nn.functional.relu(x)

        return x


class Respath(torch.nn.Module):
    '''
    ResPath

    Arguments:
        num_in_filters {int} -- Number of filters going in the respath
        num_out_filters {int} -- Number of filters going out the respath
        respath_length {int} -- length of ResPath

    '''

    def __init__(self, num_in_filters, num_out_filters, respath_length=1):

        super().__init__()

        self.respath_length = respath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        for i in range(self.respath_length):
            if (i == 0):
                self.shortcuts.append(
                    Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size=(1, 1), activation='None'))
                self.convs.append(
                    Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size=(3, 3), activation='relu'))


            else:
                self.shortcuts.append(
                    Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size=(1, 1), activation='None'))
                self.convs.append(
                    Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size=(3, 3), activation='relu'))

            self.bns.append(torch.nn.BatchNorm2d(num_out_filters))

    def forward(self, x):

        for i in range(self.respath_length):
            shortcut = self.shortcuts[i](x)

            x = self.convs[i](x)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)

            x = x + shortcut
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)

        return x


def test():
    x = torch.randn((1, 64, 256, 256))
    rp1 = Respath(num_in_filters=64, num_out_filters=64, respath_length=2)

    preds = rp1(x)

    print(preds.shape)

# if __name__ == "__main__":
#     test()
