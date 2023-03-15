import torch.nn as nn
import torch
import numpy as np
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from pytorch_msssim import ssim
from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio
from torchvision.models import vgg16

import torch
import torchvision


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        blocks = [torchvision.models.vgg16(pretrained=True).features[:4].eval(),
                  torchvision.models.vgg16(pretrained=True).features[4:9].eval(),
                  torchvision.models.vgg16(pretrained=True).features[9:16].eval(),
                  torchvision.models.vgg16(pretrained=True).features[16:23].eval()]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).to(self.device)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1).to(self.device)
            target = target.repeat(1, 3, 1, 1).to(self.device)

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class MyLoss(nn.Module):

    def __init__(self):
        super(MyLoss, self).__init__()

        # pred: model output
        # target: ground-truth

    def forward(self, target, pred):
        # MSE
        mse = F.mse_loss(input=pred, target=target)

        # LSSIM
        ssim_loss = 1 - structural_similarity_index_measure(target=target, preds=pred, data_range=1.0)

        # Perceptual Loss
        # ploss_net = VGGPerceptualLoss()
        # ploss = ploss_net(target, pred)

        # mse + lssim + perceptual loss
        return mse + ssim_loss  # + (0.1 * ploss)


def test():
    pred = torch.randn((1, 3, 400, 400)).cuda()
    gt = torch.randn((1, 3, 400, 400)).cuda()

    loss_net = MyLoss()

    loss = loss_net(gt, pred)
    print(loss)


# if __name__ == "__main__":
#     test()
