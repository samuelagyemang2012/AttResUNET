import os
from metrics.metrics import do_metrics
import time
from utils import load_checkpoint
from models.model import *
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2

separate = False
# load model
weights_path = "../res/net7_snow_large/best_snow_large_ploss_vgg19_net7_checkpoint.pth.tar"
deblur_weights_path = "../res/deblur_large/best_deblur_large_myloss_ploss_vgg19_net7_checkpoint.pth.tar"

w, h = 400, 400

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

images_folder_path = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/test2/deg/"
dest_folder_path = "C:/Users/Administrator/Desktop/datasets/snow100k/preds/snow/"

hazy_images = os.listdir(images_folder_path)

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


def batch_inference():
    inf_times = []
    for i in tqdm(hazy_images):
        input_ = Image.open(images_folder_path + i).convert('RGB')
        input_ = transform(input_)
        input_ = input_.unsqueeze(0)

        # do inference
        if not separate:
            net.eval()
            start = time.time()
            preds = net(input_)
            end = time.time()

            inf_time = get_inf_time(end, start)
            inf_times.append(inf_time)
        else:
            net.eval()
            start = time.time()
            clean = net(input_)
            preds = refine_net(clean)
            end = time.time()

            inf_time = get_inf_time(end, start)
            inf_times.append(inf_time)

        preds = process_tensor(preds)
        cv2.imwrite(dest_folder_path + i, preds * 255)
        cv2.waitKey(-1)

    avg_inf = sum(inf_times) / len(inf_times)
    print('avg inf time: {:.2f}s'.format(avg_inf))


if __name__ == "__main__":
    clear_imgs_path = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/test2/clear/"
    print("translating images")
    batch_inference()
    print("getting metrics")
    do_metrics(clear_imgs_path, dest_folder_path)
