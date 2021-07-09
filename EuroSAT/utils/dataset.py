import os
import torch
import config
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
# torch.backends.cudnn.deterministic= True
# torch.backends.cudnn.benchmark= False
import numpy as np
import config
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
BATCH_SIZE = 16
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image


def plot_imeges(data,targets):
    LABELS = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop',
              'Residential', 'River', 'SeaLake']
    fig = plt.figure(figsize=(15, 10))
    for i in range(BATCH_SIZE):
        tensor_image = data[i]
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(tensor_image.permute(1, 2, 0))
        # plt.title(paths[i].split('/')[-2])
        plt.tight_layout()
        ax.set_title("{}".format(LABELS[targets[i]]))
    plt.show()
    plt.savefig(f"output/EuroSAT_Resutls.png")



def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _, _ in loader:
        data = data.float()
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches+=1
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches-mean**2)**0.5
    return mean, std
#

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
        image = np.array(Image.open(os.path.join(self.images_folder, image_file))) # albumentations also i divided for 255
        # image = TF.to_tensor(Image.open(os.path.join(self.images_folder, image_file))) #Torch Transform
        path= os.path.join(self.images_folder, image_file)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label, path






#Test dataset, get mean and std or plot albumentation resutls
if __name__ == "__main__":
    train_dataset = GetDataset(
        images_folder='data/EuroSAT/',
        path_to_csv='../data/train.csv',
        transform=config.train_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, num_workers=2, shuffle=True, pin_memory=True)
    # mean, std = get_mean_std(train_loader)
    # print(mean)
    # print(std)
    loop = tqdm(train_loader)
    for batch_idx, (data, targets, paths) in enumerate(loop):
        # data = data.permute(0, 3, 1, 2)
        data = data.float()
        print(data.shape)
        print(data)
        print(type(data.long()))

        plot_imeges(data, targets)
        break