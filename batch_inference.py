import os

import torch
from utils import load_checkpoint
from models.model import AttResUNET2
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2

# load model
with_lrelu = True

weights_path_lr = "res/train6/best_dehaze_mse_lssim_ploss3_no_clahe_checkpoint.pth.tar"
# weights_path = "res/train5/best_dehaze_mse_lssim_ploss2_no_clahe_checkpoint.pth.tar"

w, h = 400, 400

net_lr = AttResUNET2()
weights_lr = torch.load(weights_path_lr)
net_lr = load_checkpoint(weights_lr, net_lr)

# net = AttResUNET()
# weights = torch.load(weights_path)
# net = load_checkpoint(weights, net)

images_folder_path = "C:/Users/Administrator/Desktop/datasets/SOTs/Detection/foggy road scenes.v1i.yolov5pytorch/test/images/"
dest_lr_folder_path = "C:/Users/Administrator/Desktop/datasets/SOTs/Detection/foggy road scenes.v1i.yolov5pytorch/test/preds/"
dest_folder_path = "C:/Users/Administrator/Desktop/datasets/SOTs/indoor/preds_default_lr/"

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
        net_lr.eval()
        # net.eval()

        preds_lr = net_lr(input_)
        # preds = net(input_)

        preds_lr = process_tensor(preds_lr)
        # preds = process_tensor(preds)

        cv2.imwrite(dest_lr_folder_path + i, preds_lr)
        # cv2.imwrite(dest_folder_path + i, preds)
        cv2.waitKey(-1)


if __name__ == "__main__":
    batch_inference()
