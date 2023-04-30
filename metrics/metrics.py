import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity
from tqdm import tqdm


def get_SSIM(image1, image2, is_multichannel=True):
    if is_multichannel:
        score, _ = structural_similarity(image1, image2, full=True, multichannel=True)
        return score
    else:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        score, _ = structural_similarity(image1, image2, full=True)
        return score


def get_psnr(image1, image2, max_value=255):
    mse = np.mean((np.array(image1, dtype=np.float32) - np.array(image2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def do_metrics(clear_images_path, preds_images_path, w=400, h=400):
    # clear_images_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/outdoor/clear/"
    # preds_images_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/outdoor/preds_7_bnorm_data_large/"
    w, h = 400, 400
    clear_images = os.listdir(clear_images_path)
    preds_images = os.listdir(preds_images_path)

    ssims = []
    psnrs = []

    for i in tqdm(range(len(clear_images))):
        img1 = cv2.imread(clear_images_path + clear_images[i])
        img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)

        img2 = cv2.imread(preds_images_path + preds_images[i])
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

        ssim = get_SSIM(img1, img2)
        psnr = get_psnr(img1, img2)

        ssims.append(ssim)
        psnrs.append(psnr)

    avg_ssim = sum(ssims) / len(ssims)
    avg_psnr = sum(psnrs) / len(psnrs)

    print("SSIM: {:.3f}".format(avg_ssim))
    print("PSNR: {:.2f}".format(avg_psnr))


if __name__ == "__main__":
    clear_images_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/outdoor/clear/"
    preds_images_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/outdoor/preds_7_bnorm_data_large/"
    do_metrics(clear_images_path, preds_images_path, w=256, h=256)
#     metrics()
