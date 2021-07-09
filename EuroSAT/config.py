import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import random
MODEL_LIST= ["VGG11", "VGG19", "RESNET18", "RESNET152","EFFICIENTNET-B1", "EFFICIENTNET-B7"]
MODEL_NAME= "VGG11"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
BATCH_SIZE = 256
NUM_EPOCHS = 100
NUM_WORKERS = 2
WEIGHT_DECAY = 1e-5
SEED= 42
SAVE_MODEL = True
LOAD_MODEL = False
CHECKPOINT_FILE = f"models/EuroSAT_{MODEL_NAME}_C.pth.tar"  # Save model
RESULTS_FILE = f"output/csv/EuroSAT_{MODEL_NAME}_C.csv"         # Save outputs in CSV


# Data Preparation
train_transforms = A.Compose(
    [
        A.Resize(width=64, height=64),
        # Geometry & Topology
        # A.HorizontalFlip(p=0.25),
        # A.VerticalFlip(p=0.25),
        # A.RandomRotate90(p=0.25),
        # A.Rotate(limit=random.randint(1, 89), p=0.25),
        # A.Transpose(p=0.25),
        # A.ShiftScaleRotate(p=0.25),
        # A.GridDistortion(p=0.25),
        # A.ElasticTransform(p=0.25),

        # Color
        A.RandomBrightnessContrast(p=0.25),
        A.RandomGamma(p=0.25),
        A.Blur(p=0.25),
        A.RGBShift(p=0.25),
        A.Sharpen(p=0.25),
        A.RandomFog(p=0.25),
        A.ColorJitter(p=0.25),
        A.ToFloat(max_value=None),


        A.Normalize(
            mean = [0.3445, 0.3803, 0.4078],
            std = [0.2041, 0.1369, 0.1149],
            max_pixel_value=1,), # to see original data this should be removed

        ToTensorV2(),
    ]
)


val_test_transforms = A.Compose(
    [
        A.Resize(width=64, height=64),
        A.ToFloat(max_value=None),
        A.Normalize(
            mean = [0.3445, 0.3803, 0.4078],
            std = [0.2041, 0.1369, 0.1149],
            max_pixel_value=1,
        ),
        ToTensorV2(),
    ]
)