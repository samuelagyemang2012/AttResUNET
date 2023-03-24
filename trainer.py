import numpy as np
import pandas as pd
import os
import torch.nn as nn
import torch
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import albumentations as A
from models.model import AttResUNET, XNet, AttResUNET2, XNet2
from dataset_cv import Haze
from early_stopping import EarlyStopping
from model_checkpoint import ModelCheckpoint
from lion_pytorch import Lion
from loss import MyLoss
import cv2
from utils import save_checkpoint
from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio
from utils import load_checkpoint
from configs import train_config as cfg
import gc

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


def save_img(arr, path):
    arr = process_tensor(arr)
    arr = cv2.normalize(arr, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(path, arr)


def save_results(arr, path):
    df = pd.DataFrame(arr,
                      columns=["Epoch", "Train_Loss", "Train_SSIM", "Train_PSNR", "Val_Loss", "Val_SSIM", "Val_PSNR"])
    df.to_csv(path, index=False)


def save_model(net, opt, path):
    checkpoint = {
        "state_dict": net.state_dict(),
        "optimizer": opt.state_dict(),
    }
    save_checkpoint(checkpoint, filename=path)


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

    save_path = init(cfg.RES_DIR)

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

    info = []

    if not cfg.LOAD_MODEL:
        print("creating model")
        # net = AttResUNET(in_channels=3, out_channels=3).to(cfg.DEVICE)
        # net = R2AttUNet(in_channels=3, out_channels=3, t=1).to(cfg.DEVICE)
        # net = XNet(in_channels=3, out_channels=3).to(cfg.DEVICE)
        net = XNet2(in_channels=3, out_channels=3).to(cfg.DEVICE)
    else:
        print("loading checkpoint")
        weights_path = "res/train4/best_dehaze_mse_lssim_ploss2_no_clahe_checkpoint.pth.tar"
        net = AttResUNET()
        weights = torch.load(weights_path)
        net = load_checkpoint(weights, net)
        net = net.to(cfg.DEVICE)

    print("preparing data")
    train_dataset = Haze(clear_imgs_dir=cfg.TRAIN_CLEAR_DIR, hazy_imgs_dir=cfg.TRAIN_HAZY_DIR)
    val_dataset = Haze(clear_imgs_dir=cfg.VAL_CLEAR_DIR, hazy_imgs_dir=cfg.VAL_CLEAR_DIR)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)

    print("setting criterion and opt")
    criterion = MyLoss().to(cfg.DEVICE)
    optimizer = Lion(net.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    print("setting callbacks")
    early_stopping = EarlyStopping(tolerance=cfg.TOLERANCE, metric="accuracy")
    model_checkpoint = ModelCheckpoint(metric="accuracy")

    print("begin training")

    for epoch in range(0, cfg.NUM_EPOCHS):
        # training
        net.train()
        for iteration_train, (img_clear, img_haze) in enumerate(
                tqdm(train_loader, desc="Epoch " + str(epoch + 1) + " Train")):
            # load image
            img_clear = img_clear.to(cfg.DEVICE)
            img_haze = img_haze.to(cfg.DEVICE)

            # do prediction
            clean_image = net.forward(img_haze)

            train_loss = criterion(img_clear, clean_image)
            train_ssim = structural_similarity_index_measure(target=img_clear, preds=clean_image, data_range=1.0)
            train_psnr = peak_signal_noise_ratio(target=img_clear, preds=clean_image)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # torch.nn.utils.clip_grad_norm(net.parameters(), 0.1)

            train_step_loss.append(train_loss.cpu().detach().item())
            train_step_ssim.append(train_ssim.cpu().detach().item())
            train_step_psnr.append(train_psnr.cpu().detach().item())

        # evaluation
        gc.collect()
        net.eval()
        for iteration_val, (img_clear, img_haze) in enumerate(
                tqdm(val_loader, desc="Epoch " + str(epoch + 1) + " Val")):
            img_clear = img_clear.to(cfg.DEVICE)
            img_haze = img_haze.to(cfg.DEVICE)

            clean_image = net.forward(img_haze)

            val_loss = criterion(img_clear, clean_image)
            val_ssim = structural_similarity_index_measure(target=img_clear, preds=clean_image, data_range=1.0)
            val_psnr = peak_signal_noise_ratio(target=img_clear, preds=clean_image)

            val_step_loss.append(val_loss.cpu().detach().item())
            val_step_ssim.append(val_ssim.cpu().detach().item())
            val_step_psnr.append(val_psnr.cpu().detach().item())

        # save image at interval
        if epoch % cfg.INTERVAL == 0:
            save_img(clean_image, "test/dehaze_" + str(epoch) + ".jpg", )

        # save information
        train_epoch_loss.append(sum(train_step_loss) / len(train_step_loss))
        train_epoch_ssim.append(sum(train_step_ssim) / len(train_step_ssim))
        train_epoch_psnr.append(sum(train_step_psnr) / len(train_step_psnr))

        val_epoch_loss.append(sum(val_step_loss) / len(val_step_loss))
        val_epoch_ssim.append(sum(val_step_ssim) / len(val_step_ssim))
        val_epoch_psnr.append(sum(val_step_psnr) / len(val_step_psnr))

        info.append([epoch,
                     train_epoch_loss[epoch],
                     train_epoch_ssim[epoch],
                     train_epoch_psnr[epoch],
                     val_epoch_loss[epoch],
                     val_epoch_ssim[epoch],
                     val_epoch_psnr[epoch]])

        # log information
        print(
            'Epoch: {}/{} | Train Loss: {:.6f} | Train SSIM: {:.6f} | Train PSNR: {:.6f} | Val Loss: {:.6f} | Val '
            'SSIM: {:.6f} | Val PSNR: {:.6f} '.format(epoch + 1, cfg.NUM_EPOCHS,
                                                      train_epoch_loss[epoch],
                                                      train_epoch_ssim[epoch],
                                                      train_epoch_psnr[epoch],
                                                      val_epoch_loss[epoch],
                                                      val_epoch_ssim[epoch],
                                                      val_epoch_psnr[epoch]
                                                      ))

        save_results(info, save_path + "/results.csv")

        # model checkpoint
        if model_checkpoint(val_epoch_psnr[epoch]):
            name = save_path + "/best_ploss2_xnet_checkpoint.pth.tar"
            print("Val psnr improved from {:.4f} to {:.4f}".format(model_checkpoint.get_last_best(),
                                                                   val_epoch_psnr[epoch]))
            save_model(net, optimizer, name)

        # early stopping
        if early_stopping(val_epoch_psnr[epoch]):
            name = save_path + "/last_ploss2_xnet_checkpoint.pth.tar"
            save_model(net, optimizer, name)

            print("Early Stopping on epoch {}".format(epoch + 1))
            break


if __name__ == "__main__":
    train()
