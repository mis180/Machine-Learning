import os
import torch
import config
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from osgeo import gdal


def plot_imeges(data,targets):
    LABELS= ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop',
              'Residential', 'River', 'SeaLake']
    BANDS = ['Aerosols (60m, 443nm)', 'Blue (10m, 490nm)', 'Green (10m, 560nm)', 'Red (10m, 665nm)', 'Red edge 1 (20m, 705nm)',
             'Red edge 2 (20m, 740nm)', 'Red edge3 (20m, 783nm)', 'NIR (10m, 842nm)',
                'Red edge 4 (20m, 865nm)','Water vapor (60m, 945nm)', 'Cirrus (60m, 1375nm)', 'SWIR 1 (20m, 1610nm)', 'SWIR 2 (20m, 2190nm)']
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("{}".format(LABELS[targets]), fontsize=16)
    for i in range(13):
        tensor_image = data[i]
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(tensor_image, cmap='gray')
        plt.tight_layout()
        ax.set_title("{}".format(BANDS[i]))
    plt.savefig(f"../output/EuroSATAllBands_Dataset1.png")
    #plt.show()



class GetDataset(Dataset):
    def __init__(self, images_folder, path_to_csv, train=True, transform=None):
        super().__init__()
        self.data = pd.read_csv(path_to_csv)
        self.images_folder = images_folder
        self.image_files = os.listdir(images_folder)
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.data.shape[0] if self.train else len(self.image_files)

    def __getitem__(self, index):
        if self.train:
            im_id, image_file, label, label_name = self.data.iloc[index]
        path = os.path.join(self.images_folder, image_file)
        inDs = gdal.Open(path)
        bands = [inDs.GetRasterBand(i) for i in range(1, inDs.RasterCount + 1)]
        image = np.array([band.ReadAsArray() for band in bands]).astype('float32')
        image = np.moveaxis(image, 0, -1)
        path = os.path.join(self.images_folder, image_file)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label, path

# Test dataset, get mean and std or plot albumentation resutls
if __name__ == "__main__":
    train_dataset = GetDataset(
        images_folder=r'../data/EuroSATallBands/',
        path_to_csv=r'../data/EuroSATallBands/train.csv',
        transform=config.train_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, num_workers=2, shuffle=True, pin_memory=True)
    loop = tqdm(train_loader)
    for batch_idx, (data, targets, paths) in enumerate(loop):
        data = data[15]
        targets = targets[15] # Select differen class
        data = data.squeeze()
        plot_imeges(data, targets) # Plot dataset
        break

