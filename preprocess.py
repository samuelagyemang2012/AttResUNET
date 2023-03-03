import cv2
import os
import numpy as np
from tqdm import tqdm

hazy_folder_path = "F:/datasets/SOTs/outdoor_CLAHE/hazy/"
clahe_dest_path = "F:/datasets/SOTs/outdoor_CLAHE/clahe_sharp/"


def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])

    # image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    smoothed = cv2.GaussianBlur(image, (9, 9), 10)
    unsharped = cv2.addWeighted(image, 1.5, smoothed, -0.5, 0)
    return unsharped


def do_clahe(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)

    # create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    #  apply clahe on luminosity channel
    clahe_img = clahe.apply(l)

    # merge clahe channel
    merge_img_clahe = cv2.merge((clahe_img, a, b))

    # covert final image to BGR
    img_ = cv2.cvtColor(merge_img_clahe, cv2.COLOR_LAB2BGR)

    img_ = cv2.detailEnhance(img_, sigma_s=1, sigma_r=0.15)
    img_ = cv2.edgePreservingFilter(img_, flags=1, sigma_s=64, sigma_r=0.2)

    return img_


sharpen = True

hazy_files = os.listdir(hazy_folder_path)

for hf in tqdm(hazy_files):
    arr = cv2.imread(hazy_folder_path + hf)
    if sharpen:
        arr = sharpen_image(arr)

    arr = do_clahe(arr)

    cv2.imwrite(clahe_dest_path + hf, arr)
