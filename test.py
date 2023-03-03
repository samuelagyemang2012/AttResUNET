import os
import collections

folder_path = "D:/Datasets/SOTs/data/val/hazy/"
files = os.listdir(folder_path)
a = []

for f in files:
    x = f.split("_")
    name = x[0]

    new_name = name + ".jpg"

    print(f, new_name)

    os.rename(folder_path + f, folder_path + new_name)
