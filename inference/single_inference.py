import torch
from utils import load_checkpoint, process_tensor
from models.model import *
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from metrics import metrics

# load model


print("one model")
weights_path = "../res/haze/net7_sots/best_net7_haze_sots_ploss_vgg19_checkpoint.pth.tar"
net = Network7Haze(dropout=0.2, use_batchnorm=True)
weights = torch.load(weights_path)
net = load_checkpoint(weights, net)

# "C:/Users/Administrator/Desktop/datasets/dehaze/reside/OTS/training_data/val/clear/8004.jpg"
# "C:/Users/Administrator/Desktop/datasets/snow100k/light_snow/synthetic/crossing_03084.jpg"
# C:/Users/Administrator/Desktop//datasets/dehaze/reside/SOTs/training_data/SOTS/train/hazy/0003.jpg"
# C:/Users/Administrator/Desktop/datasets/snow100k/light_snow/test/gt/city_read_01081.jpg

img_clear_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/training_data/SOTS/val/clear/1917.png"
img_deg_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/training_data/SOTS/val/hazy/1917.jpg"

transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor()
])


# def process_tensor(tensor):
#     tensor = tensor.detach().squeeze(0).permute(1, 2, 0).numpy()
#     tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
#     return tensor


def resize(arr, w, h):
    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    return arr


def single_inference():
    gt = Image.open(img_clear_path).convert('RGB')
    gt = transform(gt)
    gt = gt.unsqueeze(0)

    input = Image.open(img_deg_path).convert('RGB')
    w, h = input.size
    input = transform(input)
    input = input.unsqueeze(0)

    # do inference
    net.eval()
    start = time.time()
    preds = net(input)
    end = time.time()

    preds = process_tensor(preds)
    gt = process_tensor(gt)
    input = process_tensor(input)

    # print(gt)
    ssim = metrics.get_SSIM(gt, preds, is_multichannel=True)
    psnr = metrics.get_psnr(gt, preds, max_value=1)
    inf_time = end - start

    print("inf_time:", inf_time)
    print("ssim:", ssim)
    print("psnr:", psnr)

    cv2.imshow("groundtruth", gt)
    cv2.imshow("deg", input)
    cv2.imshow("prediction", preds)
    # cv2.imshow("z", z)
    # cv2.imshow("wb", wb)
    cv2.waitKey(-1)


if __name__ == "__main__":
    single_inference()
