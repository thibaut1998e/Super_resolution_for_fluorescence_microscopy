# Convert .lif files in 'folderIn' files in .tif files and write them in 'forlderOut' in separate channels
# Assumptions: In the .lif image series, even numbers correspond to deconvolved images and odd numbers correspond to raw images. The deconvolved images are coded in 16 bits, the raw images are coded in 8 bits
# The maximum intensity projection MIP is computed for the deconvolved images and written in separate folders for each channel


#sys.path.append('/Users/Denis/Seafile/Code/Downloaded/readlif-0.2.1')

import numpy as np
import imageio
from PIL import Image

import os
import read_lif as rl
import sys

#_, folderIn, folderOut = sys.argv

# In and out folders
#folderIn = "C:/Users/Thibaut/Downloads/Assembly"
folderIn = "/data/Eloy/widefield_ExM"
folderOut = "/home/eloy/wide_field_cell_images"

#folderIn = 'C:/Users/Thibaut/Documents/_Stage_Image_Super_Resolution/denis_thunder/denis_thunder'
#folderOut = 'C:/Users/Thibaut/Documents/_Stage_Image_Super_Resolution/denis_thunder/tiff'


if not os.path.exists(folderOut+'/deconv/c1'):
    os.makedirs(folderOut+'/deconv/c1')
if not os.path.exists(folderOut+'/deconv/c2'):
    os.makedirs(folderOut+'/deconv/c2')
if not os.path.exists(folderOut+'/deconv/mip_c1'):
    os.makedirs(folderOut+'/deconv/mip_c1')
if not os.path.exists(folderOut+'/deconv/mip_c2'):
    os.makedirs(folderOut+'/deconv/mip_c2')
if not os.path.exists(folderOut+'/raw/c1'):
    os.makedirs(folderOut+'/raw/c1')
if not os.path.exists(folderOut+'/raw/c2'):
    os.makedirs(folderOut+'/raw/c2')

nb_channels = 1


for filename in os.listdir(folderIn):
    if filename.endswith(".lif") and filename[0] != ".":
        print(filename)
        reader = rl.Reader(os.path.join(folderIn, filename))
        lifImgSeries = reader.getSeries()
        idImg = 0
        idFn = 1
        for lifImg in lifImgSeries:
            print(len(lifImg.getChannels()))
            if len(lifImg.getChannels()) == 1:
                idImg = idImg + 1;
                if idImg%2==0: # Deconv
                    # Save tif volumes
                    for c in range(nb_channels):
                        imgC = lifImg.getFrame(T=0,channel=c,dtype=np.uint16)
                        imageio.mimwrite(os.path.join(folderOut, f'deconv/c{c+1}/',
                                                      os.path.splitext(filename)[0] + '_' + str(idFn) + '.tiff'), imgC)

                        mipC1 = np.amax(imgC, axis=0).astype('float32')
                        mipC1 = ((mipC1 - mipC1.min()) / (mipC1.max() - mipC1.min()) * 255).astype('uint8')
                        im = Image.fromarray(mipC1)
                        im = im.convert("L")
                        im.save(os.path.join(folderOut, 'deconv/mip_c1/',
                                             os.path.splitext(filename)[0] + '_' + str(idFn) + '.png'))
                    #imgC2 = lifImg.getFrame(T=0,channel=1,dtype=np.uint16)
                    #imageio.mimwrite(os.path.join(folderOut, 'deconv/c1/', os.path.splitext(filename)[0] + '_' + str(idFn) + '.tiff'),imgC1)
                    #imageio.mimwrite(os.path.join(folderOut, 'deconv/c2/', os.path.splitext(filename)[0] + '_' + str(idFn) + '.tiff'),imgC2)
                    # Save MIP
                    idFn = idFn + 1;
                else: # Raw
                    for c in range(nb_channels):
                        imgC = lifImg.getFrame(T=0,channel=c,dtype=np.uint16)
                        imageio.mimwrite(os.path.join(folderOut, f'raw/c{c+1}/',
                                                      os.path.splitext(filename)[0] + '_' + str(idFn) + '.tiff'), imgC)


