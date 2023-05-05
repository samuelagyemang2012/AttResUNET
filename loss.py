import torch.nn as nn
import torch
import numpy as np
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
# from pytorch_msssim import ssim
from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio
from torchvision.models import vgg16, vgg19
import torch
import torchvision


class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerceptualLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss) / len(loss)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        blocks = [torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT').features[:4].eval(),
                  torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT').features[4:9].eval(),
                  torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT').features[9:16].eval(),
                  torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT').features[16:23].eval()]
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


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vgg = vgg19(pretrained=True).features[:35].eval().to(self.device)

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        vgg_input_features = self.vgg(pred)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)


class MyLoss(nn.Module):

    def __init__(self):
        super(MyLoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.vgg19loss = VGGLoss()
        self.vgg16loss = VGGPerceptualLoss()

        # self.vgg_model = vgg16(weights="VGG16_Weights.DEFAULT").features[:16]
        # self.vgg_model = self.vgg_model.to(self.device)
        # for param in self.vgg_model.parameters():
        #     param.requires_grad = False

        # self.ploss = VGGPerceptualLoss()
        # self.ploss = PerceptualLoss(self.vgg_model)

        # pred: model output
        # target: ground-truth

    def forward(self, target, pred):
        # MSE
        mse = F.mse_loss(input=pred, target=target)

        # LSSIM
        ssim_loss = 1 - structural_similarity_index_measure(target=target, preds=pred, data_range=1.0)

        # Perceptual Loss
        pl = self.vgg16loss(pred, target)

        # mse + lssim + perceptual loss
        return mse + ssim_loss + (0.3 * pl)


class MSESSIM(nn.Module):

    def __init__(self):
        super(MSESSIM, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, target, pred):
        # MSE
        mse = F.mse_loss(input=pred, target=target)

        # LSSIM
        ssim_loss = 1 - structural_similarity_index_measure(target=target, preds=pred, data_range=1.0)

        # mse + lssim + perceptual loss
        return mse + ssim_loss


def test():
    pred = torch.randn((1, 3, 400, 400)).cuda()
    gt = torch.randn((1, 3, 400, 400)).cuda()

    loss_net = MSESSIM()

    loss = loss_net(gt, pred)
    print(loss)


# if __name__ == "__main__":
#     test()
