import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import cv2


class SOTS(Dataset):
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
            hazy = Image.open(hazy_img_path)
            clean = Image.open(clean_img_path)
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
            img = np.array(Image.open(f).convert('RGB')).astype(np.float32)  # .astype('uint8')
            return img

    def __len__(self):
        return self.size


def test():
    def process(arr):
        arr = arr.permute(0, 1, 2).numpy()
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        arr = arr.astype('uint8')

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

    TRAIN_IMG_DIR = "C:/Users/Administrator/Desktop/datasets/SOTs/data/SOTS/train/hazy/"
    TRAIN_MASK_DIR = "C:/Users/Administrator/Desktop/datasets/SOTs/data/SOTS/train/clear/"
    VAL_IMG_DIR = "C:/Users/Administrator/Desktop/datasets/SOTs/data/SOTS/val/hazy/"
    VAL_MASK_DIR = "C:/Users/Administrator/Desktop/datasets/SOTs/data/SOTS/val/clear/"

    train_dataset = SOTS(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform)
    loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=0)

    examples = next(iter(loader))
    hazy_images, clear_images = examples

    for i, b in enumerate(hazy_images):
        print(b.shape)

        img1 = process(b)
        img2 = process(clear_images[i])

        show_images(img1, img2)


# if __name__ == "__main__":
#     test()
