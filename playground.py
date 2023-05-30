from utils import load_checkpoint, process_tensor
from models.model import *
import torch
from utils import load_checkpoint
from models.model import *
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from metrics import metrics
import time


def load_haze_net(weights_path):
    net = Network7Haze(dropout=0.2, use_batchnorm=True)
    weights = torch.load(weights_path)
    net = load_checkpoint(weights, net)
    return net


def load_snow_net(weights_path):
    net = Network7Snow(dropout=0.2, use_batchnorm=True)
    weights = torch.load(weights_path)
    net = load_checkpoint(weights, net)
    return net


def load_image(image_path, image_dim=400):
    transform = transforms.Compose([
        transforms.Resize((image_dim, image_dim)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)

    return img


def inference(net, image_tensor, show_result=True, save_image=False):
    net.eval()
    start = time.time()
    preds = net(image_tensor)
    end = time.time()

    inf_time = end - start
    print('Inference time:', inf_time)

    if show_result:
        cv2.imshow("before", process_tensor(image_tensor))
        cv2.imshow("after", process_tensor(preds))
        cv2.waitKey(-1)


if __name__ == "__main__":
    weights_path = "res/snow/net7/best_net7_snow_ploss_vgg19_checkpoint.pth.tar"
    image_path = "C:/Users/Administrator/Desktop/snow5.JPG"

    net = load_snow_net(weights_path)

    tensor = load_image(image_path)

    inference(net, tensor)
