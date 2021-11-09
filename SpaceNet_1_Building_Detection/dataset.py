import os
import config
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from osgeo import gdal
from numpy import load
from tqdm import tqdm
from torch.utils.data import DataLoader


class GetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        inDs = gdal.Open(img_path)
        bands = [inDs.GetRasterBand(i) for i in range(1, inDs.RasterCount + 1)]
        image = np.array([band.ReadAsArray() for band in bands]).astype('float32')
        image = np.moveaxis(image, 0, -1)

        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".tif", ".npy"))
        mask = load(mask_path)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


# Test dataset, get mean and std or plot albumentation resutls
if __name__ == "__main__":
    train_dataset = GetDataset(

        image_dir=r'data/rio_train/',
        mask_dir=r'data/rio_mask/',
        transform=config.train_transform)


    train_loader = DataLoader(dataset=train_dataset, batch_size=16, num_workers=2, shuffle=True, pin_memory=True)
    loop = tqdm(train_loader)
    for batch_idx, (data, mask) in enumerate(loop):
        print(data.shape)
        print(mask.shape)
        break