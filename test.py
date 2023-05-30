import os
import json
import shutil
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import cv2
from PIL import Image
import torchvision.transforms as transforms

import metrics.metrics

"""
Rename files
"""


def rename_files():
    folder_path = "F:/datasets/dehaze/OTS/hazy_real/"
    files = os.listdir(folder_path)
    a = []

    for f in files[0:3]:
        x = f.split("_")
        name = x[0]
        new_name = name + ".jpg"

        os.rename(folder_path + f, folder_path + new_name)


def get_foggy_images_bdd():
    files = []
    json_path = "F:/datasets/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    f = open(json_path, )
    data = json.load(f)

    print("data loaded")
    for d in data:
        name = d["name"]
        weather = d["attributes"]["weather"]
        time = d["attributes"]["timeofday"]
        labels = d["labels"]

        if weather == "foggy" and time == "daytime":
            for ll in labels:
                try:
                    cat = ll['category']
                    box = ll['box2d']
                    x1 = box['x1']
                    y1 = box['y1']
                    x2 = box['x2']
                    y2 = box['y2']

                    files.append([name, cat, x1, y1, x2, y2])
                except:
                    pass

    # Closing file
    f.close()

    label_map = {"traffic sign": 0,
                 "car": 1,
                 "person": 2,
                 "traffic light": 3,
                 "rider": 4,
                 "bike": 5,
                 "bus": 6,
                 "truck": 7,
                 "motor": 8,
                 "train": 9
                 }

    df = pd.DataFrame(files, columns=["file", "category", "x1", "y1", "x2", "y2"], index=None)
    df["id"] = df["category"]
    df["id"] = df["id"].map(label_map)

    df.to_csv("foggy_scenes.csv")


def move_foggy_images_bdd():
    csv_file = "foggy_scenes.csv"
    images_path = "F:/datasets/bdd100k_images_100k/bdd100k/images/100k/train/"
    iter = os.scandir(images_path)

    images = []
    for entry in iter:
        if entry.is_file():
            images.append(entry.name)

    c = 0
    #
    print("csv loaded")
    df = pd.read_csv(csv_file)
    files = df.file.to_list()

    for f in files:
        if f in images:
            pass
            # print(images_path + f)
            # print("./det/" + f)
            # shutil.copy(images_path + f, "./det/images/" + f)
    # print(c)


def assert_data():
    csv_path = "foggy_scenes.csv"
    det_images_path = "./det/images/"

    df = pd.read_csv(csv_path)
    det_images = os.listdir(det_images_path)

    assert len(df["file"].unique()) == len(det_images)
    print("same length: {}, {}".format(len(det_images), len(df["file"].unique())))


def write(data, path):
    f = open(path, "w")
    f.write(data)
    f.close()


def to_yolo(w_, h_, box):
    dw = 1. / float(w_)
    dh = 1. / float(h_)
    ll = box[4].astype(int)
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = round(x * dw, 4)
    w = round(w * dw, 4)
    y = round(y * dh, 4)
    h = round(h * dh, 4)
    # print(type(ll))
    return ll.astype(str), x.astype(str), y.astype(str), w.astype(str), h.astype(str)


def get_labels():
    csv_path = "foggy_scenes.csv"
    w, h = (1280, 720)

    df = pd.read_csv(csv_path, )

    images = df["file"].unique()

    for i in tqdm(images):
        data = df[df["file"] == i]
        x = data.to_numpy()
        string = ""

        for d in x:
            id, x1, y1, x2, y2 = d[6], d[2], d[3], d[4], d[5]
            new_box = np.array([x1, x2, y1, y2, id])
            id, x, y, w_, h_ = to_yolo(w, h, new_box)

            string += "{} {} {} {} {}".format(id, x, y, w_, h_) + "/n"
        write(string, "./det/anns/" + str(i) + ".txt")


def its():
    hazy_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/ITS/hazy/"
    dest_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/ITS/hazy_copy/"

    hazy_files = os.listdir(hazy_path)

    for hf in tqdm(hazy_files):
        arr = hf.split("_")
        c = arr[1]

        if c == "10":
            shutil.copy(hazy_path + hf, dest_path + hf)
            # print(new_name)


def split():
    images_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/ITS/hazy_copy/"

    train_clear_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/ITS/training_data/train/clear/"
    train_hazy_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/ITS/training_data/train/hazy/"

    val_clear_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/ITS/training_data/val/clear/"
    val_hazy_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/ITS/training_data/val/hazy/"

    source_clear_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/ITS/clear/"
    source_hazy_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/ITS/hazy_copy/"

    images = os.listdir(images_path)
    data = []

    for i in images:
        data.append(i)

    random.shuffle(data)

    train_split = int(len(data) * 0.8)

    train_data = data[0:train_split]
    val_data = data[train_split:]

    for td in train_data:
        cl = train_clear_path + td
        shutil.copy(source_clear_path + td, cl)

        hz = train_hazy_path + td
        shutil.copy(source_hazy_path + td, hz)

    for vd in val_data:
        cl = val_clear_path + vd
        shutil.copy(source_clear_path + vd, cl)

        hz = val_hazy_path + vd
        shutil.copy(source_hazy_path + vd, hz)


def temperature(image, temp):
    kelvin_table = {
        1000: (255, 56, 0),
        1500: (255, 109, 0),
        2000: (255, 137, 18),
        2500: (255, 161, 72),
        3000: (255, 180, 107),
        3500: (255, 196, 137),
        4000: (255, 209, 163),
        4500: (255, 219, 186),
        5000: (255, 228, 206),
        5500: (255, 236, 224),
        6000: (255, 243, 239),
        6500: (255, 249, 253),
        7000: (245, 243, 255),
        7500: (235, 238, 255),
        8000: (227, 233, 255),
        8500: (220, 229, 255),
        9000: (214, 225, 255),
        9500: (208, 222, 255),
        10000: (204, 219, 255)}

    x = Image.fromarray(np.uint8(image * 255))
    r, g, b = kelvin_table[temp]
    matrix = (r / 255.0, 0.0, 0.0, 0.0,
              0.0, g / 255.0, 0.0, 0.0,
              0.0, 0.0, b / 255.0, 0.0)
    return x.convert('RGB', matrix)


# Reading the image


def ots():
    images_path = "F:/datasets/dehaze/OTS/clear/hazy/part4/part4/"
    dest_path = "F:/datasets/dehaze/OTS/clear/hazy/part4/haze1.02/"

    iter = os.scandir(images_path)

    for entry in tqdm(iter):
        if entry.is_file():
            filename = entry.name
            arr = filename.split("_")
            end = arr[-1].split(".jpg")
            x = arr[1] + " " + end[0]

            if x == "1 0.2":
                b = x.replace(" ", "_")
                src = images_path + filename
                dest = dest_path + arr[0] + "_" + b + ".jpg"

                shutil.copy(src, dest)
                # print(src)
                # print(dest)
                # print("")


def compare_images():
    from metrics.metrics import get_SSIM
    c = 0
    set_1 = "F:/datasets/dehaze/OTS/clear/clear/"
    set_2 = "F:/datasets/dehaze/SOTS-Test/outdoor/clear/"

    hazy_path = "F:/datasets/dehaze/OTS/clear/hazy_real/"
    dest_base_path = "F:/datasets/dehaze/OTS/clear/misc/"

    set1_files = os.listdir(set_1)
    hazy_files = os.listdir(hazy_path)

    set2_files = os.listdir(set_2)

    for s2 in set2_files:
        for i, s1 in enumerate(set1_files):
            a = s2.split(".")[0]
            b = s1.split(".")[0]

            if a == b:
                clear_src = set_1 + s1
                clear_dest = dest_base_path + "clear/" + s1

                hazy_src = hazy_path + hazy_files[i]
                hazy_dest = dest_base_path + "hazy/" + hazy_files[i]

                shutil.move(clear_src, clear_dest)
                shutil.move(hazy_src, hazy_dest)

                # print("clear_: ", set_1 + s1)
                # print("hazy: ", hazy_path + hazy_files[i])


def snow100k():
    clear_path = "F:/datasets/Snow100K/all/gt/"
    snow_path = "F:/datasets/Snow100K/all/synthetic/"

    train_clear_dest = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/train/clear/"
    train_deg_dest = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/train/deg/"

    val_clear_dest = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/val/clear/"
    val_deg_dest = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/val/deg/"

    test_clear_dest = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/test/clear/"
    test_deg_dest = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/test/deg/"

    test2_clear_dest = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/test2/clear/"
    test2_deg_dest = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/test2/deg/"

    gt = os.listdir(snow_path)
    random.shuffle(gt)

    train = gt[0:1500]
    val = gt[1500:1900]
    test = gt[1900:2400]

    for tr in tqdm(train):
        clear_img_path = clear_path + tr
        snow_img_path = snow_path + tr

        clear_dest_path = train_clear_dest + tr
        snow_dest_path = train_deg_dest + tr

        shutil.copy(clear_img_path, clear_dest_path)
        shutil.copy(snow_img_path, snow_dest_path)

    for vl in tqdm(val):
        clear_img_path = clear_path + vl
        snow_img_path = snow_path + vl

        clear_dest_path = val_clear_dest + vl
        snow_dest_path = val_deg_dest + vl

        shutil.copy(clear_img_path, clear_dest_path)
        shutil.copy(snow_img_path, snow_dest_path)

    for tt in tqdm(test):
        clear_img_path = clear_path + tt
        snow_img_path = snow_path + tt

        clear_dest_path = test_clear_dest + tt
        snow_dest_path = test_deg_dest + tt

        shutil.copy(clear_img_path, clear_dest_path)
        shutil.copy(snow_img_path, snow_dest_path)


def snow100k_test():
    clear_source_path = "F:/datasets/Snow100K/all/gt/"
    deg_source_path = "F:/datasets/Snow100K/all/synthetic/"

    train_path = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/train/deg/"
    val_path = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/val/deg/"
    test_path = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/test/deg/"

    clear_dest_path = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/test2/clear/"
    deg_dest_path = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/test2/deg/"

    clear_source_files = os.listdir(clear_source_path)
    # deg_source_files = os.listdir(deg_source_path)

    train_files = os.listdir(train_path)
    val_files = os.listdir(val_path)
    test_files = os.listdir(test_path)

    for tr in train_files:
        if tr in clear_source_files:
            clear_source_files.remove(tr)

    for vl in val_files:
        if vl in clear_source_files:
            clear_source_files.remove(vl)

    for tt in test_files:
        if tt in clear_source_files:
            clear_source_files.remove(tt)

    print(len(clear_source_files))

    random.shuffle(clear_source_files)
    test2_files = clear_source_files[0:500]

    for tf in tqdm(test2_files):
        clear_source = clear_source_path + tf
        deg_source = deg_source_path + tf

        clear_dest = clear_dest_path + tf
        deg_dest = deg_dest_path + tf

        shutil.copy(clear_source, clear_dest)
        shutil.copy(deg_source, deg_dest)


def intersection():
    x_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/OTS/training_data/val/clear/"
    y_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/OTS/training_data/train/clear/"

    x_files = os.listdir(x_path)
    y_files = os.listdir(y_path)

    res = set(x_files).intersection(set(y_files))
    print(len(res))
    return list(res)


def process_tensor(tensor):
    tensor = tensor.detach().squeeze(0).permute(1, 2, 0).numpy()
    tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
    return tensor


def get_image_quality(path_img1, path_img2, dim=400):
    img1 = Image.open(path_img1).convert('RGB')
    img2 = Image.open(path_img2).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((dim, dim)),
        transforms.ToTensor()
    ])

    img1 = transform(img1)
    img1 = img1.unsqueeze(0)

    img2 = transform(img2)
    img2 = img2.unsqueeze(0)

    img1 = process_tensor(img1)
    img2 = process_tensor(img2)

    ssim = metrics.metrics.get_SSIM(img1, img2, is_multichannel=True)
    psnr = metrics.metrics.get_psnr(img1, img2, max_value=1.0)

    print("ssim: ", ssim)
    print("psnr: ", psnr)


if __name__ == "__main__":
    path1 = "C:/Users/Administrator/Desktop/city_read_11796.jpg"
    path2 = "C:/Users/Administrator/Desktop/city_read_11796_ged.jpg"
    get_image_quality(path1, path2)

    # intersection()
    # snow100k_test()
    # compare_images()
    # ots()
    # split()
    # rename_files()
    # get_labels()
    # get_foggy_images_bdd()
    # move_foggy_images_bdd()
    # assert_data()
    # its()
    # white_balance()
