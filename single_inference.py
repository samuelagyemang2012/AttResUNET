import torch
from utils import load_checkpoint
from models.model import AttResUNET, AttResUNET2, XNet, XNet2
from PIL import Image
import torchvision.transforms as transforms
import cv2

# load model
weights_path = "C:/Users/Administrator/Desktop/dehaze/res/xnet_no_final_con_its_bs_1/best_ploss2_xnet_no_fs_checkpoint.pth.tar"

net = XNet2()  # AttResUNET() #XNet()  # AttResUNET2()
weights = torch.load(weights_path)
net = load_checkpoint(weights, net)

img_clear_path = "./test/0469.jpg" #"C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/training_data/SOTS/val/clear/1917.png"  # "C:/Users/Administrator/Downloads/HSTS/real-world/NW_Google_837.jpeg"  # "C:/Users/Administrator/Desktop/datasets/SOTs/data_no_clahe/SOTS/val/clear/1919.png"
img_hazy_path ="./test/0469.jpg" #"C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/training_data/SOTS/val/hazy/1917.jpg"  # "C:/Users/Administrator/Downloads/HSTS/real-world/NW_Google_837.jpeg"

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

    input = Image.open(img_hazy_path).convert('RGB')
    w, h = input.size
    input = transform(input)
    input = input.unsqueeze(0)

    # do inference
    net.eval()
    preds = net(input)
    preds = process_tensor(preds)
    # preds = resize(preds, w, h)

    input = process_tensor(input)
    # input = resize(input, w, h)

    cv2.imshow("groundtruth", gt)
    cv2.imshow("hazy", input)
    cv2.imshow("prediction", preds)
    cv2.waitKey(-1)


if __name__ == "__main__":
    single_inference()
