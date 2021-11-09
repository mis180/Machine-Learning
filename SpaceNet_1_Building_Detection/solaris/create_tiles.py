import solaris as sol
import os

img_path = r'C:\PycharmProjects\semantic_segmentation_unet\data\sn2_AOI_2_Vegas\sn2_SN2_buildings_train_AOI_2_Vegas_PS-RGB_img1\PS-RGB.tif'
gjason_path= r''


raster_tiler = sol.tile.raster_tile.RasterTiler(dest_dir=r'C:\PycharmProjects\semantic_segmentation_unet\data\vegas_train',  # the directory to save images to
                                                src_tile_size=(500, 500),  # the size of the output chips
                                                verbose=True)


raster_bounds_crs = raster_tiler.tile(img_path)


vector_tiler = sol.tile.vector_tile.VectorTiler(dest_dir=r'/data/rio_label',
                                                verbose=True)
vector_tiler.tile('C:/Users/Meir/PycharmProjects/Solaris/Rio_Buildings_Public_AOI_v2.geojson',
                  tile_bounds=raster_tiler.tile_bounds,
                  tile_bounds_crs=raster_bounds_crs)