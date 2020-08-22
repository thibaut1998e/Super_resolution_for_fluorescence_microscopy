import topaz_commands as tp
from fastai.vision import *

import libtiff

import skimage
import skimage.io
import skimage.filters
from fastai.script import *

from scipy.ndimage.interpolation import zoom as npzoom

import scipy.ndimage as scp

#from fastai.script import *
import paths_definitions as pth
from utils.multi import MultiImage
import transformations as tf
import plot_graphs as pg
import apply_transformation as aptf

#torch.cuda.device(0)


def inference(learner_name, lr_path=pth.wide_field_3D, scale=2, hr_folder='', lr_folder='', test_on_wide_field=True, test_on_training=True,
              raw=False, center_txt=None, topaz=False, topaz2=False, size=None):

    """make prediction on wide field 3D images and on validation data and store results in a folder.

    learner_name, str : name without extension of the .pkl model. It should be located in the folder pth.models
    lr_path, str : path of 3D test images. Processed images are located in {lr_path}/deconv/c2 if raw=False and at
    {lr_path}/raw/c2 otherwise

    lr_folder, hr_folder : name of HR folder and LR folder used for training the model. Processed images are located
    in {training_sets}/{lr_folder}/{relative}/valid with relative = 'deconv/c2' if not raw and 'raw/c2' otherwise. HR images from hr folder
    are also saved in the results folder to visualize quickly the results

    test_on_wide_field, boolean : predict on wide field data
    test_on_training : predict on validation data

    topaz, boolean : if true prediction is made only on patches centered on centrioles, other pixels are set to 0
    topaz2, boolean  : if true generate as many output images as centrioles in input image (one for each)

    centre_txt, str : should be provided if topaz or topaz2. path of the txt file computed by topaz which contains the
    centers of particles in lr_path (wide field data)

    scale, int : used to compute bilinnear interpolation of images in lr_folder. Also used if topaz = True to determine the
    shape of the output image

    size : used if topaz or topaz2 : size of the patches cropped in LR wide field images."""
    #test_directory = myHome + "/test_results"

    learn = get_learner(learner_name)
    relative = pth.relative_path_type_channel if not raw else pth.relative_path_raw
    lr_path = lr_path + "/" + relative
    results = pth.test_results_dir + "/" + learner_name
    if os.path.exists(results):
        shutil.rmtree(results)
    os.makedirs(results)
    lr_path_val = f'{pth.training_sets}/{lr_folder}/{relative}/valid'
    if test_on_training:
        print(f'testing on validation data at location : {lr_path_val}')
    hr_path_val = f'{pth.training_sets}/{hr_folder}/{relative}/valid'
    results_validation = f'{results}/test_on_training' #folder which contains results on validation data
    test_files_validation = os.listdir(lr_path_val)[:10]
    pg.plot_loss(learner_name) #plot validation losses and metrics and save the graphs
    os.makedirs(f'{results_validation}/{pth.LR}')
    os.makedirs(f'{results_validation}/{pth.HR}')
    os.makedirs(f'{results_validation}/{pth.HR_predicted}')
    os.makedirs(f'{results_validation}/{pth.HR_bilinear}')
    if test_on_wide_field:
        print(f'testing on wide field data at location : {lr_path}')
    results_test = f'{results}/test_on_wide_field' #folder which contains results on wide field data
    test_files = os.listdir(lr_path)
    os.makedirs(results_test)
    center_dict = {}
    if topaz or topaz2:
        center_dict = tp.get_center_dict_from_txt(center_txt=center_txt, threshold=0)
    print('wide field', test_on_wide_field)
    print('training', test_on_training)
    if test_on_wide_field:
        print('test_files',test_files)
        print(len(test_files))
        for fn in test_files:

            predict_and_save(learn, fn, results_test, lr_path, wide_field=True
                                       ,scale=scale, topaz=topaz, topaz2=topaz2,
                             center_dict=center_dict, tile_size_topaz=size)
    if test_on_training:
        for fn in test_files_validation:
            predict_and_save(learn, fn, results_validation, lr_path_val, hr_path_val, scale=scale)

    print(f'results saved at location : {results}')

def predict_and_save(learn, file_name, results, lr_path, hr_path='', scale=2,  topaz=False,
                     topaz2=False, wide_field=False, center_dict=None, tile_size_topaz=None):

    """predict an hr image for imaga at location {lr_path}/file_name. If test_on_wide_field is False it also saves HR, HR_bilinaear and LR
    images in other subfolders
    topaz : if true prediction is made only on patches centered on centrioles, other pixels are set to 0
    topaz2 : if true generate as many output images as centrioles in input image (one for each)"""
    data = libtiff.TiffFile(Path(lr_path+"/"+file_name))
    data = data.get_tiff_array()
    img = data[:].astype(np.float32)
    print(f'process image : {lr_path}/{file_name}, shape : {data.shape}')
    if img.shape[0] == 1: #2D image
        img = img[0]
        out_imgs = prediction(learn, img)
    else: #3D image
        if topaz:
            im_name = file_name.split('.')[0]
            out_imgs = np.array([prediction_tiles_topaz(learn, img[i], f'{im_name}_slice_{i}.tiff', scale=scale,
                                                        center_dict=center_dict, tile_size_topaz=tile_size_topaz)
                                for i in range(len(img))])
        elif topaz2:
            im_name = file_name.split('.')[0]
            out_imgs = reconstruction_3D_centrioles(learn, img, im_name, center_dict=center_dict, tile_size_topaz=tile_size_topaz)
        else:
            # predict on each slices and stack them
            out_imgs = predict_3D_image(learn, img)

    if not isinstance(out_imgs, list):
        out_imgs = [out_imgs]
    #out_img = unet_image_from_tiles_blend(learn, img, tile_sz=size, scale = scale, img_info=img_info)
    for i,out_img in enumerate(out_imgs):
        out_img = out_img.astype(np.float32)
        name = f'{file_name.split(".")[0]}_{i}.tiff' if i >= 1 else file_name
        if not wide_field:
            skimage.io.imsave(results + "/" + pth.HR_predicted + "/" + name, out_img)
            skimage.io.imsave(results + "/" + pth.LR + "/" + name, np.array(data))
            resized_with_interpolation = scp.zoom(np.array(data), scale, order=3)
            skimage.io.imsave(results + "/" + pth.HR_bilinear + "/" + name, resized_with_interpolation)
            real_hr_image = np.array(skimage.io.imread(hr_path + "/" + name))
            skimage.io.imsave(results + "/" + pth.HR + "/" + name, real_hr_image)
        else:
            skimage.io.imsave(results + "/" + name, out_img)


def get_learner(learner_name):
    return load_learner(pth.models, file=f'{learner_name}.pkl').to_fp32()


def prediction_tiles_topaz(learn, in_img, im_name, center_dict=None, tile_size_topaz=50, scale=2):
    """predict only on patches centered on centrioles, other pixels are set to 0. The mean of the predctions is computed in the
    pixels on which patches overlap
    tile_size_topaz : size of patches in LR images
    in_img : 2D array"""
    out_image = np.zeros((in_img.shape[0] * scale, in_img.shape[1] * scale))
    h, w = out_image.shape
    tiles = tf.crop_with_center_dict_2(in_img, center_dict=center_dict, tile_sz=tile_size_topaz, image_name=im_name)
    X_min = {}; X_max = {}; Y_min = {}; Y_max = {}
    out_tiles = {}
    for c in tiles.keys():
        out_tile = prediction(learn, tiles[c])
        out_tiles[c] = out_tile
        X_min[c] = max(scale*c[0]-tile_size_topaz, 0)
        X_max[c] = min(scale*c[0]+tile_size_topaz, h)
        Y_min[c] = max(scale*c[1]-tile_size_topaz, 0)
        Y_max[c] = min(scale*c[1]+tile_size_topaz, w)
    for i in range(h):
        for j in range(w):
            cpt = 0
            for c in tiles.keys():
                if i >= X_min[c] and i < X_max[c] and j >= Y_min[c] and j < Y_max[c]:
                    out_image[i][j] += out_tiles[c][i-X_min[c]][j-Y_min[c]]
                    cpt += 1
                if cpt > 1:
                    out_image[i][j] /= cpt

    return out_image


def prediction(learn, in_img):
    """in_img is a 2D array, returns the image 2D array predicted by the model"""
    in_img = tf.normalize(in_img)
    in_img = tensor(in_img)
    pred = learn.model(in_img[None][None])
    # add 2 dimensions because input of fastai models are 4 dimensional arrays of shape (batch size, channels, weight, height)
    # if in_img shape is (156, 156) then in_img[None][None] shape is (1,1,156,156)
    out_array = np.array(pred)[0][0] #output are also 4d arrays with shape (1,1,312,312)
    return out_array


def reconstruction_3D_centrioles(learn, in_img, im_name, center_dict=None, tile_size_topaz=50, radius=30, cut=True):
    """returns as many output images as centrioles in the input images.
    center_dict is the dictionary computed with the txt file from topaz : keys are names and values list of centers.
    This dictionary contains centers predicted for all the slices.
    tile_size_topaz is the size of the patches cropped in LR images.
    The position of the center of one 3D centriole is computed as the average over all the slices of its position.
    If 2 centers predicted in 2 different slices are closer than radius, they are consired to be the center of the
    same centriole.
    If cut is True, it cuts in the depth dimension at the position of local minima of intensity, to separate centrioles"""
    centrioles = tf.crop_threeD_with_center_dict(in_img, tile_sz=tile_size_topaz, center_dict=center_dict, cut=cut,
                                                 radius=radius, image_name=im_name, nb_of_crop=2)
    return [predict_3D_image(learn, tile) for tile in centrioles]


def predict_3D_image(learn, in_img):
    #predict on each slices and stack them
    return np.array([prediction(learn, slice) for slice in in_img])





