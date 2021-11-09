import os
from os import listdir
from os.path import isfile, join
import shutil
import solaris as sol
import skimage
import geopandas as gpd
from matplotlib import pyplot as plt
from shapely.ops import cascaded_union
import numpy as np
from osgeo import gdal
from numpy import save


img_dir=r"C:\PycharmProjects\semantic_segmentation_unet\data\sn1_AOI_1_RIO"
mask_dir  =r'C:\PycharmProjects\semantic_segmentation_unet\data\rio_mask'
train_dir = r"C:\PycharmProjects\semantic_segmentation_unet\data\rio_train"
label_dir =r"C:\PycharmProjects\semantic_segmentation_unet\data\rio_label"

subfolders = [ f.path for f in os.scandir(img_dir) if f.is_dir() ]
img_folder=[]
for i in  subfolders:
    j = i.split('-')[-1]
    if j !="labels":
        img_folder.append(i)

for img_dir in img_folder:
    #reading
    img_path = os.path.join(img_dir,"RGB.tif")
    main_name = img_dir.split('RIO\\')[-1]
    label_name = img_dir+'-labels'
    label_path = os.path.join(label_name,"labels.geojson")
    #computing
    gdf = gpd.read_file(label_path)
    cascaded_union(gdf.geometry.values)
    mask = sol.vector.mask.footprint_mask(df=label_path, reference_im=img_path)
    #writing
    img_name =main_name+".tif"
    mask_name=  main_name+".npy"
    label_name2=  main_name+".geojson"
    img_path2 = os.path.join(train_dir ,img_name)
    mask_path = os.path.join(mask_dir,mask_name)
    label_path2=  os.path.join(label_dir,label_name2)
    #saving and moving
    save(mask_path, mask)
    shutil.move(img_path, img_path2)
    shutil.move(label_path, label_path2)
    print(img_path2)