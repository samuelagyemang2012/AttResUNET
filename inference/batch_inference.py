import os
from metrics.metrics import do_metrics
import time
from utils import load_checkpoint
from models.model import *
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
from metrics import metrics
from configs import train_config as cfg

separate = False
# load model
weights_path = "../res/haze/net7_sots/best_net7_haze_sots_ploss_vgg19_checkpoint.pth.tar"

w, h = 400, 400
net = Network7Haze(use_batchnorm=True)
weights = torch.load(weights_path)
net = load_checkpoint(weights, net)

# "C:/Users/Administrator/Desktop/datasets/snow100k/light_snow/test/syn/
# "C:/Users/Administrator/Desktop/datasets/snow100k/light_snow/test/gt/"
# "C:/Users/Administrator/Desktop/datasets/snow100k/light_snow/test/preds/snow_l/"
# "C:/Users/Administrator/Desktop/datasets/dehaze/reside/OTS/training_data/val/hazy/"
# "C:/Users/Administrator/Desktop/datasets/dehaze/reside/OTS/preds/haze/"
# "C:/Users/Administrator/Desktop/datasets/dehaze/reside/OTS/training_data/val/clear/"

images_folder_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/training_data/SOTS/val/hazy/"
gt_images_folder_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/training_data/SOTS/val/clear/"
dest_folder_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/prediction/SOTS_outdoor/"

deg_images = os.listdir(images_folder_path)
gt_images = os.listdir(gt_images_folder_path)

transform = transforms.Compose([
    transforms.Resize((w, h)),
    transforms.ToTensor()
])


def process_tensor(tensor):
    tensor = tensor.detach().squeeze(0).permute(1, 2, 0).numpy()
    tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
    # tensor = cv2.normalize(tensor, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return tensor


def resize(arr, w, h):
    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    return arr


def get_inf_time(end, start):
    return end - start


def get_quality_metrics(img1, img2, is_multichannel=True, max_value=1.0):
    ssim = metrics.get_SSIM(img1, img2, is_multichannel=is_multichannel)
    psnr = metrics.get_psnr(img1, img2, max_value=max_value)

    return ssim, psnr


def write_data(data, path):
    f = open(path, "w")
    f.write(data)
    f.close()


def batch_inference():
    inf_times = []
    ssims = []
    psnrs = []

    save_img = True

    for i, k in enumerate(tqdm(deg_images)):

        input_ = Image.open(images_folder_path + k).convert('RGB')
        input_ = transform(input_)
        input_ = input_.unsqueeze(0)

        gt_ = Image.open(gt_images_folder_path + gt_images[i]).convert('RGB')
        gt_ = transform(gt_)
        gt_ = gt_.unsqueeze(0)

        # do inference
        net.eval()
        start = time.time()
        preds = net(input_)
        end = time.time()

        inf_time = get_inf_time(end, start)
        inf_times.append(inf_time)

        preds = process_tensor(preds)
        gt = process_tensor(gt_)

        # cv2.imshow("pred", preds)
        # cv2.imshow("gt", gt)
        # cv2.waitKey(-1)

        ssim, psnr = get_quality_metrics(gt, preds)
        ssims.append(ssim)
        psnrs.append(psnr)

        if save_img:
            cv2.imwrite(dest_folder_path + k, preds * 255)

    avg_inf = sum(inf_times) / len(inf_times)
    avg_ssim = sum(ssims) / len(ssims)
    avg_psnr = sum(psnrs) / len(psnrs)

    print('avg inf time: {:.2f}s'.format(avg_inf))
    print('avg SSIM: {:.2f}'.format(avg_ssim))
    print('avg PSNR: {:.2f}'.format(avg_psnr))

    data = "avg inf time: {:.2f}s".format(avg_inf) + "\n" + \
           "avg SSIM: {:.2f}".format(avg_ssim) + "\n" + \
           "avg PSNR: {:.2f}".format(avg_psnr)

    filename = "00_" + dest_folder_path.split("/")[-2] + ".txt"
    write_data(data, dest_folder_path + filename)


if __name__ == "__main__":
    batch_inference()
