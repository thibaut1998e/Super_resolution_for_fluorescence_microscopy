import os
import shutil
import sys

_, folder, invertList = sys.argv

#folder = '/Users/Denis/Documents/data/superres_centriole/Assembly_tif_clean'
#invertList ='/Users/Denis/Documents/data/superres_centriole/invertList.txt'

invertListFile = open(invertList, 'r')
Lines = invertListFile.readlines()
for line in Lines:
    print(folder+'/deconv/c1/'+line[:-1]+'.tiff')
    if os.path.exists(folder+'/deconv/c1/'+line[:-1]+'.tiff'):
        print("bjr")
        shutil.move(folder+'/deconv/c1/'+line[:-1]+'.tiff',folder+'/deconv/'+line[:-1]+'.tiff')
        shutil.move(folder+'/deconv/c2/'+line[:-1]+'.tiff',folder+'/deconv/c1/'+line[:-1]+'.tiff')
        shutil.move(folder+'/deconv/'+line[:-1]+'.tiff',folder+'/deconv/c2/'+line[:-1]+'.tiff')

        shutil.move(folder+'/raw/c1/'+line[:-1]+'.tiff',folder+'/raw/'+line[:-1]+'.tiff')
        shutil.move(folder+'/raw/c2/'+line[:-1]+'.tiff',folder+'/raw/c1/'+line[:-1]+'.tiff')
        shutil.move(folder+'/raw/'+line[:-1]+'.tiff',folder+'/raw/c2/'+line[:-1]+'.tiff')
