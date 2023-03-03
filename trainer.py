import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
import albumentations as A
from model import AttResUNET
from dataset import SOTS
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
from early_stopping import EarlyStopping
from lion_pytorch import Lion
from loss import MyLoss
from tqdm import tqdm

# Hyperparameters etc.
DATASET = SOTS
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 2
NUM_WORKERS = 0
IMAGE_HEIGHT = 400  # 1280 originally
IMAGE_WIDTH = 400  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "D:/Datasets/SOTs/data/train/hazy/"
TRAIN_MASK_DIR = "D:/Datasets/SOTs/data/train/clear/"
VAL_IMG_DIR = "D:/Datasets/SOTs/data/val/hazy/"
VAL_MASK_DIR = "D:/Datasets/SOTs/data/val/clear/"


def train():
    # train_transform = A.Compose(
    #     [
    #         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #         A.Rotate(limit=35, p=1.0),
    #         A.HorizontalFlip(p=0.5),
    #         A.VerticalFlip(p=0.1),
    #         A.Normalize(
    #             # mean=[0.0, 0.0, 0.0],
    #             # std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )

    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        # A.Normalize(
        #     mean=[0.0, 0.0, 0.0],
        #     std=[1.0, 1.0, 1.0],
        #     max_pixel_value=255.0,
        # ),
        ToTensorV2()
    ])

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Normalize(
            #     mean=[0.0, 0.0, 0.0],
            #     std=[1.0, 1.0, 1.0],
            #     max_pixel_value=255.0,
            # ),
            ToTensorV2(),
        ],
    )

    print("loading model")
    net = AttResUNET(in_channels=3, out_channels=3).to(DEVICE)

    print("preparing data")
    train_dataset = SOTS(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform)
    val_dataset = SOTS(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=val_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    print("setting criterion and opt")
    criterion = MyLoss()
    optimizer = Lion(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(tolerance=5, min_delta=10)
    net.train()

    print("begin training")
    for epoch in range(0, NUM_EPOCHS):
        for iteration_train, (img_orig, img_haze) in enumerate(train_loader):
            # load image
            img_orig = img_orig.to(DEVICE)
            img_haze = img_haze.permute(0, 3, 1, 2).to(DEVICE)

            # do prediction
            clean_image = net(img_haze)

            train_loss = criterion(clean_image, img_orig)

            optimizer.zero_grad()
            train_loss.backward()
            # torch.nn.utils.clip_grad_norm(dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()

        for iteration_val, (img_orig, img_haze) in enumerate(val_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image = net(img_haze)

            val_loss = criterion(clean_image, img_orig)

        print("Epoch " + str(epoch) + ": train_loss: " + str(train_loss.item() + " val_loss: " + str(val_loss.item())))

        early_stopping(train_loss.item(), val_loss.item())
        if early_stopping.early_stop:
            print("Early Stopping")


if __name__ == "__main__":
    train()
