import os
import torch
import numpy as np
import config
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image
import random
randnum= random.randint(1, 89)

#Geometrical_Topologocal_Transformation

names =["Horizontal Flip",
        "Vertical Flip",
        "Random Rotate 90",
        "Rotate, select random degree: {}".format(randnum),
        "Transpose",
        "Shift Scale Rotate",
        "Grid Distortion",
        "Elastic Transformation",
         ]

arr = [ A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.RandomRotate90(p=1),
        A.Rotate(limit=randnum),
        A.Transpose(p=1),
        A.ShiftScaleRotate(p=1),
        A.GridDistortion(p=1),
        A.ElasticTransform(p=1,alpha=1, sigma=50),
       ]



# Color_Transformation

# names =["RandomBrightnessContrast",
#         "RandomGamma",
#         "Blur",
#         "RGBShift: {}".format(randnum),
#         "Sharpen",
#         "RandomFog",
#         "ColorJitter",
#          ]
#
# arr = [ A.RandomBrightnessContrast(p=1),
#         A.RandomGamma(p=1),
#         A.Blur(p=1),
#         A.RGBShift(p=1),
#         A.Sharpen(p=1),
#         A.RandomFog(p=1),
#         A.ColorJitter(p=1),
#        ]



path = r"../data/EuroSAT/Industrial/Industrial_31.jpg"
im = np.array(Image.open(path))
fig = plt.figure(figsize=(10, 8))

transform = A.Compose([
    A.Resize(width=64, height=64),
    A.RandomRotate90(p=0.5),
    ToTensorV2(),
])

for i in range(-1, len(arr)):
    arr2 = [A.Resize(width=64, height=64),]
    if i > -1:
        arr2.append(arr[i])
    arr2.append(ToTensorV2())
    transform = A.Compose(arr2)
    ax = fig.add_subplot(3, 3, i + 2, xticks=[], yticks=[])
    image = transform(image=im)["image"]
    ax.imshow(image.permute(1, 2, 0))
    plt.tight_layout()
    if i > -1:
        ax.set_title("{}".format(names[i]))
    else:
        ax.set_title("Original Image")
plt.show()
