import os
import shutil


def split_train_valid(source_folder, proportion):

    types = ["deconv", "raw"]
    channels = ["c1", "c2"]
    for typ in types:
        for channel in channels:
            path = f'{source_folder}/{typ}/{channel}'
            file_names = os.listdir(path)
            train_path = f'{path}/train'
            valid_path = f'{path}/valid'
            os.makedirs(train_path)
            os.makedirs(valid_path)
            for i in range(len(file_names)):
                file_path = f'{path}/{file_names[i]}'
                if i % int(1/proportion) == 0:
                    shutil.move(file_path, valid_path)
                else:
                    shutil.move(file_path, train_path)





def split_train_valid_hr_lr_together(hr_folder, lr_folder, proportion):
    file_names = os.listdir(hr_folder)
    os.makedirs(f'{hr_folder}/train')
    os.makedirs(f'{hr_folder}/valid')
    os.makedirs(f'{lr_folder}/train')
    os.makedirs(f'{lr_folder}/valid')
    for i in range(len(file_names)):
        if i % int(1/proportion) == 0:
            shutil.move(f'{hr_folder}/{file_names[i]}', f'{hr_folder}/valid')
            shutil.move(f'{lr_folder}/{file_names[i]}', f'{lr_folder}/valid')
        else:
            shutil.move(f'{hr_folder}/{file_names[i]}', f'{hr_folder}/train')
            shutil.move(f'{lr_folder}/{file_names[i]}', f'{lr_folder}/train')



if __name__ == "__main__":
    #split_train_valid("/home/eloy/assembly_tif", 0.04)
    #split_train_valid()
    split_train_valid_hr_lr_together('/data/Eloy/data_PSSR/EM/lr_tiles_128', '/data/Eloy/data_PSSR/EM/hr_tiles_512', 0.04)





