import os
import json
import shutil
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import cv2
from PIL import Image

"""
Rename files
"""


def rename_files():
    folder_path = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/ITS/hazy_copy/"
    files = os.listdir(folder_path)
    a = []

    for f in files:
        x = f.split("_")
        name = x[0]
        new_name = name + ".png"

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


def eqHist():
    import cv2
    # Reading the image


if __name__ == "__main__":
    pass
# split()
# rename_files()
# get_labels()
# get_foggy_images_bdd()
# move_foggy_images_bdd()
# assert_data()
# its()
# white_balance()
