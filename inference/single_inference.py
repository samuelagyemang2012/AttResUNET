import torch
from utils import load_checkpoint
from models.model import *
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np

# load model
separate = False

weights_path = "../res/net7_snow_large/best_snow_large_ploss_vgg19_net7_checkpoint.pth.tar"
deblur_weights_path = "../res/deblur_large/best_deblur_large_myloss_ploss_vgg19_net7_checkpoint.pth.tar"

if not separate:
    net = Network7(in_channels=3, out_channels=3, dropout=0.2, use_batchnorm=False)
    weights = torch.load(weights_path)
    net = load_checkpoint(weights, net)
else:
    net = Network7(in_channels=3, out_channels=3, dropout=0.2, use_batchnorm=False)
    refine_net = DeBlur(use_batch=True)

    model_weights = torch.load(weights_path)
    net = load_checkpoint(model_weights, net)

    refine_weights = torch.load(deblur_weights_path)
    refine_net = load_checkpoint(refine_weights, refine_net)

img_clear_path = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data/test/clear/beautiful_smile_04354.jpg"
img_deg_path = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data/test/deg/beautiful_smile_04354.jpg"  # /datasets/dehaze/reside/SOTs/training_data/SOTS/train/hazy/0003.jpg"

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


def temperature(image, temp):
    kelvin_table = {
        1000: (255, 56, 0),
        1500: (255, 109, 0),
        2000: (255, 137, 18),
        2500: (255, 161, 72),
        3000: (255, 180, 107),
        3500: (255, 196, 137),
        4000: (255, 209, 163),
        4500: (255, 219, 186),
        5000: (255, 228, 206),
        5500: (255, 236, 224),
        6000: (255, 243, 239),
        6500: (255, 249, 253),
        7000: (245, 243, 255),
        7500: (235, 238, 255),
        8000: (227, 233, 255),
        8500: (220, 229, 255),
        9000: (214, 225, 255),
        9500: (208, 222, 255),
        10000: (204, 219, 255)}

    x = Image.fromarray(np.uint8(image * 255))
    r, g, b = kelvin_table[temp]
    matrix = (r / 255.0, 0.0, 0.0, 0.0,
              0.0, g / 255.0, 0.0, 0.0,
              0.0, 0.0, b / 255.0, 0.0)
    x = x.convert('RGB', matrix)
    x.show()


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
        preds = net(input)
    else:
        net.eval()
        clean = net(input)
        preds = refine_net(clean)

    preds = process_tensor(preds)

    input = process_tensor(input)
    # input = resize(input, w, h)

    cv2.imshow("groundtruth", gt)
    cv2.imshow("deg", input)
    cv2.imshow("prediction", preds)
    # cv2.imshow("wb", wb)
    cv2.waitKey(-1)


if __name__ == "__main__":
    single_inference()
