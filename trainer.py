import numpy as np
import os
import torch.nn as nn
import torch
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import albumentations as A
from model import AttResUNET, UNET
from dataset_cv import SOTS
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim
from early_stopping import EarlyStopping
from lion_pytorch import Lion
from loss import MyLoss
import cv2
from utils import save_checkpoint
from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio
from PIL import Image

# Hyperparameters etc.
DATASET = SOTS
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
INTERVAL = 10
NUM_EPOCHS = 50
NUM_WORKERS = 0
IMAGE_HEIGHT = 400  # 1280 originally
IMAGE_WIDTH = 400  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
RES_DIR = "res/"
TRAIN_HAZY_DIR = "C:/Users/Administrator/Desktop/datasets/SOTs/data/SOTS/train/hazy/"
TRAIN_CLEAR_DIR = "C:/Users/Administrator/Desktop/datasets/SOTs/data/SOTS/train/clear/"
VAL_HAZY_DIR = "C:/Users/Administrator/Desktop/datasets/SOTs/data/SOTS/val/hazy/"
VAL_CLEAR_DIR = "C:/Users/Administrator/Desktop/datasets/SOTs/data/SOTS/val/clear/"


def init(path):
    if len(os.listdir(path)) == 0:
        name = path + "train" + str(0)
        os.mkdir(name + "/")
        return name
    else:
        name = path + "train" + str(len(os.listdir(path)))
        os.mkdir(name + "/")
        return name


def process_tensor(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
    return tensor


def show_image(arr):
    cv2.imshow("", arr)
    cv2.waitKey(-1)


def save_img(path, arr):
    arr = cv2.normalize(arr, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(path, arr)


def train():
    # writer = SummaryWriter()

    # train_transform = A.Compose([
    #     A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #     # A.Rotate(limit=35, p=1.0),
    #     A.HorizontalFlip(p=0.5),
    #     # A.VerticalFlip(p=0.1),
    #     A.Normalize(
    #         mean=[0.0, 0.0, 0.0],
    #         std=[1.0, 1.0, 1.0],
    #         max_pixel_value=255.0,
    #     ),
    #     ToTensorV2()
    # ])

    # val_transform = A.Compose([
    #     A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #     A.Normalize(
    #         mean=[0.0, 0.0, 0.0],
    #         std=[1.0, 1.0, 1.0],
    #         max_pixel_value=255.0,
    #     ),
    #     ToTensorV2(),
    # ])

    train_step_loss = []
    train_step_ssim = []
    train_step_psnr = []

    val_step_loss = []
    val_step_ssim = []
    val_step_psnr = []

    train_epoch_loss = []
    train_epoch_ssim = []
    train_epoch_psnr = []

    val_epoch_loss = []
    val_epoch_ssim = []
    val_epoch_psnr = []

    print("loading model")
    net = AttResUNET(in_channels=3, out_channels=3).to(DEVICE)

    print("preparing data")
    train_dataset = SOTS(clear_imgs_dir=TRAIN_CLEAR_DIR, hazy_imgs_dir=TRAIN_HAZY_DIR)
    val_dataset = SOTS(clear_imgs_dir=VAL_CLEAR_DIR, hazy_imgs_dir=VAL_CLEAR_DIR)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    print("setting criterion and opt")
    criterion = MyLoss().to(DEVICE)
    optimizer = Lion(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    early_stopping = EarlyStopping(tolerance=5, min_delta=0)

    print("begin training")

    for epoch in range(0, NUM_EPOCHS):
        # training
        net.train()
        for iteration_train, (img_clear, img_haze) in enumerate(tqdm(train_loader, desc="Epoch "+str(epoch)+" Train")):
            # load image
            img_clear = img_clear.to(DEVICE)
            img_haze = img_haze.to(DEVICE)

            # do prediction
            clean_image = net.forward(img_haze)

            train_loss = criterion(img_clear, clean_image)
            train_ssim = structural_similarity_index_measure(target=img_clear, preds=clean_image, data_range=1.0)
            train_psnr = peak_signal_noise_ratio(target=img_clear, preds=clean_image)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # torch.nn.utils.clip_grad_norm(net.parameters(), 0.1)

            # print('Epoch: {}/{} | Step: {}/{} | Train SSIM: {:.6f} | Train PSNR: {:.6f} | Train Loss: {:.6f}'
            #       .format(epoch + 1, NUM_EPOCHS,
            #               iteration_train, len(train_loader),
            #               train_ssim,
            #               train_psnr,
            #               train_loss.cpu().detach().item()))

            train_step_loss.append(train_loss.cpu().detach().item())
            train_step_ssim.append(train_ssim.cpu().detach().item())
            train_step_psnr.append(train_psnr.cpu().detach().item())

        # save image at interval
        if epoch % INTERVAL == 0:
            clean_image = process_tensor(clean_image)
            save_img("test/dehaze_" + str(epoch) + ".jpg", clean_image)
            # show_image(clean_image)

        # evaluation
        net.eval()
        for iteration_val, (img_clear, img_haze) in enumerate(tqdm(val_loader, desc="Epoch "+str(epoch)+" Val")):
            img_clear = img_clear.to(DEVICE)
            img_haze = img_haze.to(DEVICE)

            clean_image = net.forward(img_haze)

            val_loss = criterion(img_clear, clean_image)
            val_ssim = structural_similarity_index_measure(target=img_clear, preds=clean_image, data_range=1.0)
            val_psnr = peak_signal_noise_ratio(target=img_clear, preds=clean_image)

            # print('Epoch: {}/{} | Step: {}/{} | Val SSIM: {:.6f} | Val PSNR: {:.6f} | Val Loss: {:.6f}'
            #       .format(epoch + 1, NUM_EPOCHS,
            #               iteration_val, len(val_loader),
            #               val_ssim,
            #               val_psnr,
            #               val_loss.cpu().detach().item()))

            val_step_loss.append(val_loss.cpu().detach().item())
            val_step_ssim.append(val_ssim.cpu().detach().item())
            val_step_psnr.append(val_psnr.cpu().detach().item())

        # print info
        train_epoch_loss.append(sum(train_step_loss) / len(train_step_loss))
        train_epoch_ssim.append(sum(train_step_ssim) / len(train_step_ssim))
        train_epoch_psnr.append(sum(train_step_psnr) / len(train_step_psnr))

        val_epoch_loss.append(sum(val_step_loss) / len(val_step_loss))
        val_epoch_ssim.append(sum(val_step_ssim) / len(val_step_ssim))
        val_epoch_psnr.append(sum(val_step_psnr) / len(val_step_psnr))

        # log information
        print(
            'Epoch: {}/{} | Train Loss: {:.6f} | Train SSIM: {:.6f} | Train PSNR: {:.6f} | Val Loss: {:.6f} | Val '
            'SSIM: {:.6f} | Val PSNR: {:.6f} '.format(epoch + 1, NUM_EPOCHS,
                                                      train_epoch_loss[epoch],
                                                      train_epoch_ssim[epoch],
                                                      train_epoch_psnr[epoch],
                                                      val_epoch_loss[epoch],
                                                      val_epoch_ssim[epoch],
                                                      val_step_psnr[epoch]
                                                      ))

        # save model
        if epoch % INTERVAL == 0:
            checkpoint = {
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename="dehaze_checkpoint.pth.tar")

        # early_stopping(train_loss.item(), val_loss.item())
        # if early_stopping.early_stop:
        #     print("Early Stopping")

        # writer.flush()


if __name__ == "__main__":
    train()
