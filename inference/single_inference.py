import torch
from utils import load_checkpoint
from models.model import *
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from metrics import metrics

# load model
separate = False

if not separate:
    print("one model")
    weights_path = "../res/train18/best_netsr_b4_sa_myloss_ploss_vgg19_net7_checkpoint.pth.tar"

    net = NetworkSR3(num_blocks=4)
    # net = Network7(use_batchnorm=False)
    # net = NetworkSR2(num_blocks=8)
    weights = torch.load(weights_path)
    net = load_checkpoint(weights, net)
else:
    weights_path = "../res/netsr2_b8/best_netsr2_b8_myloss_ploss_vgg19_net7_checkpoint.pth.tar"
    deblur_weights_path = "../res/train17/best_deblur_netL_mse_checkpoint.pth.tar"

    net = NetworkSR2(num_blocks=8)  # Network7Att(, use_batchnorm=False)
    refine_net = Network7L(in_channels=3, out_channels=3, dropout=0.2, use_batchnorm=True)

    model_weights = torch.load(weights_path)
    net = load_checkpoint(model_weights, net)

    refine_weights = torch.load(deblur_weights_path)
    refine_net = load_checkpoint(refine_weights, refine_net)

img_clear_path = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/test2/clear/winter_building_04993.jpg"
img_deg_path = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/test2/deg/winter_building_04993.jpg"  # /datasets/dehaze/reside/SOTs/training_data/SOTS/train/hazy/0003.jpg"

transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor()
])


def process_tensor(tensor):
    tensor = tensor.detach().squeeze(0).permute(1, 2, 0).numpy()
    tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
    return tensor


def resize(arr, w, h):
    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    return arr


def single_inference():
    gt = cv2.imread(img_clear_path)
    gt = resize(gt, 400, 400)

    input = Image.open(img_deg_path).convert('RGB')
    w, h = input.size
    input = transform(input)
    input = input.unsqueeze(0)

    # do inference
    if not separate:
        net.eval()
        start = time.time()
        preds = net(input)
        end = time.time()
    else:
        net.eval()

        start = time.time()
        clean = net(input)
        preds = refine_net(clean)
        end = time.time()

    preds = process_tensor(preds)

    input = process_tensor(input)
    # input = resize(input, w, h)
    ssim = metrics.get_SSIM(input, preds, is_multichannel=True)
    psnr = metrics.get_psnr(input, preds, max_value=1.0)
    inf_time = end - start

    print("inf_time:", inf_time)
    print("ssim:", ssim)
    print("psnr:", psnr)

    cv2.imshow("groundtruth", gt)
    cv2.imshow("deg", input)
    cv2.imshow("prediction", preds)
    # cv2.imshow("wb", wb)
    cv2.waitKey(-1)


if __name__ == "__main__":
    single_inference()
