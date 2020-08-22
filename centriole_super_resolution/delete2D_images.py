import os
from skimage import io
import numpy as np
import sys


_, path = sys.argv
#path = "C:/Users/Thibaut/Documents/_Stage_Image_Super_Resolution/data/assembly_tiff"




for type in ["deconv", "raw"]:
    print(type)
    for channel in ["c1", "c2"]:
        folder_path = path + "/" + type + "/" + channel
        print(channel)
        for name in os.listdir(folder_path):
            image_path = folder_path + "/" + name
            image = io.imread(image_path)
            if len(image.shape) != 3:
                os.remove(image_path)
                print(image_path)