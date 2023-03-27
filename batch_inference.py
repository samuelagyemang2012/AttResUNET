import os

import torch
from utils import load_checkpoint
from models.model import AttResUNET2, XNet, XNet2
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2

# load model
weights_path = "./res/train1/best_ploss2_xnet_checkpoint.pth.tar"

w, h = 400, 400

net = XNet2(respath_len=1)
weights = torch.load(weights_path)
net = load_checkpoint(weights, net)

# net = AttResUNET()
# weights = torch.load(weights_path)
# net = load_checkpoint(weights, net)

images_folder_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/outdoor/hazy/"
dest_folder_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/outdoor/preds_xnet_rp_1/"

hazy_images = os.listdir(images_folder_path)

transform = transforms.Compose([
    transforms.Resize((w, h)),
    transforms.ToTensor()
])


def process_tensor(tensor):
    tensor = tensor.detach().squeeze(0).permute(1, 2, 0).numpy()
    tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
    tensor = cv2.normalize(tensor, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return tensor


def resize(arr, w, h):
    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    return arr


def batch_inference():
    for i in tqdm(hazy_images):
        input_ = Image.open(images_folder_path + i).convert('RGB')
        input_ = transform(input_)
        input_ = input_.unsqueeze(0)

        # do inference
        net.eval()
        preds = net(input_)
        preds = process_tensor(preds)

        cv2.imwrite(dest_folder_path + i, preds)
        cv2.waitKey(-1)


if __name__ == "__main__":
    batch_inference()
