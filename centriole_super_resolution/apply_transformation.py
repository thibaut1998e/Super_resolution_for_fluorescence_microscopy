import os
from skimage import io
import numpy as np
import imageio
import shutil
import paths_definitions as pth
import transformations as tf
import topaz_commands as tc
import time
from pathlib import Path
import sys
from functools import partial
#from inference import get_learner





def transform(folder_in, folder_out, funcs_to_apply, folders_to_skip=[], **kwargs):
    """
    apply all the funcion in the list functs_to_apply (in the same order)
    to all the images in folder_in and store the results in folder_out. Keep the same architecture of folder_in by recursively
    calling the function on each subfolders

    transformations f in funcs_to_apply take either  2D or 3D array as input, and some arguments in kwargs,
    and return either a 2-3D array or a list of 2-3D arrays.
    transformation arguments are passed in kwargs.

    transformations wont be applied to subfolders in folders_to_skip

    if folder_in and folder_out are only names of folder (they do not start with '/', thes folder are considered to be
    located in the folder training_sets.

    Example of use:
    transform('/home/eloy/training_sets/HR/', '/home/eloy/training_sets/LR_4',
        [tf.convolution, tf.resize, tf.normalize], sigma=4, scale=2, folders_to_skip=['raw', 'c1'])


    """
    if folder_in[0] != '/':
        folder_in = f'{pth.training_sets}/{folder_in}'
    if folder_out[0] != '/':
        folder_out = f'{pth.training_sets}/{folder_out}'
    if os.path.exists(folder_out):
        shutil.rmtree(folder_out)
    os.makedirs(folder_out)
    files = [f for f in os.listdir(folder_in) if f not in folders_to_skip]
    for file in files:
        path = f'{folder_in}/{file}'
        if os.path.isdir(path):
            print(f'process folder {file}')
            transform(path, f'{folder_out}/{file}', funcs_to_apply, folders_to_skip, **kwargs)
        else:
            kwargs['image_name'] = file
            image = io.imread(path)
            im_array = np.array(image)
            for f in funcs_to_apply:
                if isinstance(im_array, list):
                    for i in range(len(im_array)):
                        im_array[i] = f(im_array[i], **kwargs)
                    im_array = flatten(im_array)
                else:
                    im_array = f(im_array, **kwargs)

            if isinstance(im_array, list):
                for i, slice in enumerate(im_array):
                    name = file.split('.')[0]
                    write_array(slice, f'{folder_out}/{name}_slice_{i}.tiff')


            else:
                write_array(im_array, f'{folder_out}/{file}')


def write_array(array, path):
    name = path.split('/')[-1]
    if isinstance(array, np.ndarray):

        if len(array.shape) == 2:
            imageio.imwrite(path, array)

        elif len(array.shape) == 3:
            imageio.mimwrite(path, array)

        else:

            print(f"can't write image {name} shape must be 2 or 3")
    else:
        print(f"image {name} is not a numpy array")




def flatten(list_of_lists):
    res = []
    for i in range(len(list_of_lists)):
        if not isinstance(list_of_lists[i], list):
            res.append(list_of_lists[i])
        else:
            for j in range(len(list_of_lists[i])):
                res.append(list_of_lists[i][j])
    return res




"""some examples of use"""


def crop_topaz(fin, fout, center_txt, tile_sz=100):
    """crop images using results from topaz"""
    dict = tc.get_center_dict_from_txt(center_txt, threshold=0, nb_center_per_slice=1)
    transform(fin, fout, [tf.crop_with_center_dict, tf.normalize], tile_sz=tile_sz, center_dict=dict)


def crop_threeD_topaz(fin, fout, center_txt):
    """crop 3D images using results from topaz"""
    dict = tc.get_center_dict_from_txt(center_txt, threshold=0, nb_center_per_slice=4)

    transform(fin, fout, [tf.crop_threeD_with_center_dict, tf.normalize], center_dict=dict, radius=20,
              tile_sz=50, cut=False, average=True, folders_to_skip=['raw'], save_figure=True, nb_of_crop=3)


def conv_resize_norm(fin, fout, sigma=4, scale=0.5, noise=False):
    """applies convolution resizing and normalization to images in fin and store them in fout"""
    if not noise:
        transform(fin, fout, [tf.convolution, tf.resize, tf.normalize], sigma=sigma, scale=scale)
    else:
        transform(fin, fout, [tf.convolution, tf.resize, tf.add_noise, tf.normalize], sigma=sigma, scale=scale, sigma_noise=0.05)


def normalize(fin, fout):
    transform(fin, fout, [tf.normalize])


def cross_section(fin, fout):
    """transforms 3D images in fin into 2D images by keeping the slice of highest intensity, then crop images so that
    they all have the same shape"""
    return transform(fin, fout, [tf.cross_section, tf.crop_center, tf.normalize], x_size=312, y_size=312,
                     folders_to_skip=['c1', 'mip_c2', 'mip_c1'])


def add_spots(fin, fout):
    """adds gaussian spots to images in fin and store them in fout"""
    transform(fin, fout, [tf.add_guassian_spot], sigma_s=[3, 1], ampl=[0.5, 0.01], freq=0.01, spot_width=10)





if __name__ == "__main__":
    """
    fin = 'assembly_tif'
    f_section = 'HR_2D_n'
    print('cross section')
    cross_section(fin, f_section)
    f_spots = 'HR_spots4'
    print('add spots')
    add_spots(f_section, f_spots)
    """
    #fin = f'{pth.myHome}/wide_field/wide_field_resized'
    fin = f'{pth.myHome}/wide_field/wide_field_cell_images'
    fout = f'{pth.myHome}/wide_field/test_crop_3D_3'
    center_txt = f'{pth.myHome}/center_particles_wide_field_cell_images.txt'

    #center_txt = f'{pth.myHome}/center_particles_wide_field_resized.txt'
    crop_threeD_topaz(fin, fout, center_txt)




























