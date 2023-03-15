import torch
from utils import load_checkpoint
from model import AttResUNET
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2

net = AttResUNET()
weights = torch.load("dehaze_checkpoint.pth.tar")
net = load_checkpoint(weights, net)

img_path = "C:/Users/Administrator/Desktop/datasets/SOTs/outdoor_CLAHE/clahe_no_sharp/0302_0.9_0.16.jpg"

transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor()
])

x=cv2.imread(img_path)
cv2.imshow("",x)
cv2.waitKey(-1)

img = Image.open(img_path).convert('RGB')
img = transform(img)

img = img.unsqueeze(0)

print(img.shape)

net.eval()
preds = net(img)
preds = preds.detach().squeeze(0).permute(1, 2, 0).numpy()

preds = preds
# preds = preds.astype(np.uint8)
# print(preds)

preds = cv2.cvtColor(preds, cv2.COLOR_BGR2RGB)

cv2.imshow("",preds)
cv2.waitKey(-1)

#
# print(preds.shape)
# pred_img = Image.fromarray((preds)).convert('RGB')
# pred_img.show()
# # print(preds)
