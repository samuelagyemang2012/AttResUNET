import os
from PIL import Image
from os.path import splitext
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torchvision
import torch
import matplotlib.pyplot as plt
import cv2


class Test(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.hazy_images = [image_dir + f for f in os.listdir(image_dir) if
                            f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

        self.clean_images = [mask_dir + f for f in os.listdir(mask_dir) if
                             f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

        # Filter the images to ensure they are counterparts of the same scene
        self.filter_files()
        self.size = len(self.hazy_images)

        self.transform = transform

    def filter_files(self):
        assert len(self.hazy_images) == len(self.clean_images)
        hazy_ims = []
        clean_ims = []

        for hazy_img_path, clean_img_path in zip(self.hazy_images, self.clean_images):

            hazy = cv2.imread(hazy_img_path)
            clean = cv2.imread(clean_img_path)

            if hazy.size == clean.size:
                hazy_ims.append(hazy_img_path)
                clean_ims.append(clean_img_path)

        self.hazy_images = hazy_ims
        self.clean_images = clean_ims

    def __getitem__(self, index):
        hazy_img = self.rgb_loader(self.hazy_images[index])
        clean_img = self.rgb_loader(self.clean_images[index])

        if self.transform is not None:
            augmentations = self.transform(image=hazy_img, mask=clean_img)
            hazy_img = augmentations["image"]
            clean_img = augmentations["mask"]

        return hazy_img, clean_img

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img).astype(np.float32)

            return img

    def __len__(self):
        return self.size


class SOTS(Dataset):
    def __init__(self, clear_imgs_dir, hazy_imgs_dir, scale=1):
        self.clear_imgs_dir = clear_imgs_dir
        self.hazy_imgs_dir = hazy_imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, "Scale must be between 0 and 1"

        self.clear_images = [
            file for file in os.listdir(clear_imgs_dir) if
            file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')
        ]

        self.hazy_images = [
            file for file in os.listdir(hazy_imgs_dir) if
            file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')
        ]

        # print(self.clear_images)

    def __len__(self):
        return len(self.clear_images)
        # return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, image_size):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, "Scale is too small"
        pil_img = pil_img.resize((image_size, image_size))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        # idx = self.ids[i]

        # mask_file = self.hazy_images + "/" + idx + ""
        hazy_file = self.hazy_imgs_dir + self.hazy_images[i]
        # print("mf", hazy_file)

        # img_file = self.imgs_dir + "/" + idx
        clear_file = self.clear_imgs_dir + self.clear_images[i]
        # print("cf", clear_file)

        hazy_img = Image.open(hazy_file)
        clear_img = Image.open(clear_file)

        # assert (
        #         clear_img.size == hazy_img.size
        # ), f"Image and mask {idx} should be the same size, but are {clear_img.size} and {hazy_img.size}"

        clear_img = self.preprocess(clear_img, self.scale, 400)
        hazy_img = self.preprocess(hazy_img, self.scale, 400)

        return (
            torch.from_numpy(clear_img).type(torch.FloatTensor),
            torch.from_numpy(hazy_img).type(torch.FloatTensor),
        )


def test():
    def process(arr):
        arr = arr.squeeze(0).permute(1, 2, 0).numpy()
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        return arr

    def show_images(hazy, clear):
        cv2.imshow("haze", hazy)
        cv2.imshow("clear", clear)
        cv2.waitKey(-1)

    train_transform = A.Compose([
        A.Resize(height=400, width=400),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        # A.Normalize(
        #     mean=[0.0, 0.0, 0.0],
        #     std=[1.0, 1.0, 1.0],
        # max_pixel_value=255.0,
        # ),
        # ToTensorV2()
    ])

    TRAIN_HAZY_DIR = "C:/Users/Administrator/Desktop/datasets/SOTs/data/SOTS/train/hazy/"
    TRAIN_CLEAR_DIR = "C:/Users/Administrator/Desktop/datasets/SOTs/data/SOTS/train/clear/"

    VAL_HAZY_DIR = "C:/Users/Administrator/Desktop/datasets/SOTs/data/SOTS/val/hazy/"
    VAL_CLEAR_DIR = "C:/Users/Administrator/Desktop/datasets/SOTs/data/SOTS/val/clear/"

    train_dataset = SOTS(clear_imgs_dir=TRAIN_CLEAR_DIR, hazy_imgs_dir=TRAIN_HAZY_DIR)  # , transform=train_transform)
    loader = DataLoader(dataset=train_dataset, batch_size=3, shuffle=True, num_workers=0)

    examples = next(iter(loader))
    clear_images, hazy_images = examples

    for i, ci in enumerate(clear_images):
        clear_image = process(ci)
        print(clear_image)

        hazy_image = process(hazy_images[i])
        print(hazy_image)

        show_images(hazy_image, clear_image)


# if __name__ == "__main__":
#     test()
