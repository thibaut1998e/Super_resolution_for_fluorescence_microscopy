import os
import sys




#folder = '/Users/Denis/Documents/data/superres_centriole/Assembly_tif_clean'

folder = 'C:/Users/Thibaut/Documents/_Stage_Image_Super_Resolution/data/assembly_tiff'
deleteList = 'C:/Users/Thibaut/Documents/_Stage_Image_Super_Resolution/PSSR-master/deleteList.txt'
deleteListFile = open(deleteList, 'r', errors='ignore')
Lines = deleteListFile.readlines()
for line in Lines:
    #print(line)
    tmp = folder+'/deconv/c1/'+line[:-1]+'.tiff'
    if os.path.exists(tmp):
        os.remove(tmp)
    tmp = folder+'/deconv/c2/'+line[:-1]+'.tiff'
    if os.path.exists(tmp):
        print(line)
        os.remove(tmp)
    tmp = folder+'/raw/c1/'+line[:-1]+'.tiff'
    if os.path.exists(tmp):
        os.remove(tmp)
    tmp = folder+'/raw/c2/'+line[:-1]+'.tiff'
    if os.path.exists(tmp):
        os.remove(tmp)
