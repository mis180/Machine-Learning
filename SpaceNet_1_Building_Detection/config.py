import torch
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Hyperparameters etc.
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 25
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 500
IMAGE_WIDTH = 500
INPUT_SIZE= 500
PIN_MEMORY = True
LOAD_MODEL = True



train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),

        # A.Resize(width=int(INPUT_SIZE + INPUT_SIZE * 0.1), height=int(INPUT_SIZE + INPUT_SIZE * 0.1)),
        # A.RandomCrop(height=INPUT_SIZE, width=INPUT_SIZE),
        # # Geometry & Topology
        # A.HorizontalFlip(p=0.25),
        # A.VerticalFlip(p=0.25),
        # A.RandomRotate90(p=0.25),
        # A.Rotate(limit=random.randint(1, 89), p=0.25),
        # A.Transpose(p=0.25),
        # A.ShiftScaleRotate(p=0.25),
        # A.GridDistortion(p=0.25),

        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)
