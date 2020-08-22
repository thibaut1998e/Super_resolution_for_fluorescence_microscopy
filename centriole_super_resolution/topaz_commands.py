import os
from skimage import io
import numpy as np
import imageio
import shutil
import paths_definitions as pth
import transformations as tf
import time
import apply_transformation as aptf


def compute_center_particles_using_trained_neunet_model(image_set, relative='',tile_sz=None, model_name='part_detection_epoch100.sav',
                                                        radius=40, centriole_size=60, thershold=-100):
    """receive an image set path of either 2D or 3D images and execute a topaz command line which computes the center
    of the particles using the model given.

    The proccessed folder is image_set/relative, it is relevant because the name of the last folder of the path of image_set
    is used to define the name of the txt saved.
    Ex : image_set = '/home/eloy/HR_n'  relative = 'deconv/c2/' txt saved : center_particles_HR_n,
    folder processed : '/home/eloy/HR_n/deconv/c2'

    First if images in image_set/relative are 3D images, a folder containing all the slices is created in {image_set}/{relative}_slices

    Then, images are rescaled so that the centriole size is 60 pixels.
    Images are divided if needed in patches of size tile_sz, to prevent GPU memory errors. If tile_sz = None, tile_sz is
    set to the size of the images.
    Then centers are computed.
    It returns the path of txt file in which results are saved.

    If the folder image_set is split in several subfolders (ex: train - valid), it will compute the centers of images in all of them

    radius : parameter r used by topaz to prevent predicting several centers for one particle : topaz will not predict 2 centers
    closer than radius pixels (in rescaled images)

    centriole_size : used to compute upscale = 60/centriole_size

    Threshold : centers with score less than threshold are not taken into account

    !!!topaz virtual env should be activated to run this!!!"""

    name_set = image_set.split('/')[-1]
    image_set = f'{image_set}/{relative}'
    print(f'processed folder : {image_set}')
    print('radius', radius)
    print('centriole size', centriole_size)

    t_start = time.time()
    upscale = 60/centriole_size
    model_path = f'{pth.models}/{model_name}'

    if os.path.isdir(f'{image_set}/{os.listdir(image_set)[0]}'):
        splits = os.listdir(image_set)
        fold = f'{image_set}/{splits[0]}'
    else:
        splits = ['']
        fold = image_set

    im_shape = np.array(io.imread(f'{fold}/{os.listdir(fold)[0]}')).shape
    if tile_sz is None:
        tile_sz = im_shape[1]
    print('shape', im_shape)

    slices = False
    if len(im_shape)==3:
        slices = True
        print('cross section')
        aptf.transform(image_set, f'{image_set}_slices', [tf.cross_section_slices], first=0, p=0)
        image_set = f'{image_set}_slices'
        im_shape = (im_shape[1], im_shape[2])
        print('temps écoulé', time.time()-t_start)

    def tiles_iterator(im_shape, tile_sz):
        for x in range(int(im_shape[0]//tile_sz+0.5)):
            for y in range(int(im_shape[1]//tile_sz+0.5)):
                x_min = tile_sz*x; x_max = tile_sz*(x+1); y_min = tile_sz*y; y_max = tile_sz*(y+1)
                yield x_min, x_max, y_min, y_max

    center_txt_scaled = f'{pth.myHome}/center_particles_{name_set}.txt'
    if os.path.exists(center_txt_scaled):
        os.remove(center_txt_scaled)
    txt = open(center_txt_scaled, 'w')
    for x_min, x_max, y_min, y_max in tiles_iterator(im_shape, tile_sz):
        print('x_min : ', x_min); print('x_max : ', x_max); print('y_min : ', y_min); print('y_max : ', y_max)
        resized = f'{pth.training_sets}/resized_{x_min}_{y_min}'

        print('resizing and crop')
        aptf.transform(image_set, resized, [tf.crop, tf.resize, tf.normalize],
                  scale=upscale, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max, order=1)
        print('temps écoulé', time.time() - t_start)
        for split in splits:
            center_txt = f'{pth.myHome}/center_particles.txt'
            cmd = 'topaz extract'
            cmd += f' -r {radius}'
            cmd += f' -m {model_path}'
            cmd += f' -o {center_txt}'
            cmd += f' {resized}/{split}/*.tiff'
            print('extract particles')
            os.system(cmd)
            print('temps écoulé', time.time() - t_start)


            with open(center_txt, 'r') as f:
                lines = f.readlines()
            os.remove(center_txt)

            #write the position of center in non-scales images by dividing each values by upscale and adding coordinate
            #of top-left corner of the patch.
            #txt.write(lines[0])
            for line in lines[1:]:
                line = line.split('\t')
                line[1] = str(int(int(line[1])//upscale + y_min))
                line[2] = str(int(int(line[2])//upscale + x_min))
                score = float(line[3].split('/')[0])
                if score >= thershold:
                    str_line = ''
                    for i,x in enumerate(line):
                        str_line += x
                        if i!=len(line)-1:
                            str_line += '\t'
                    txt.write(str_line)
        shutil.rmtree(resized)

    if slices:
        shutil.rmtree(image_set)
    txt.close()
    print(f'results saved in location : {center_txt_scaled}')

    return center_txt_scaled


def dist(c1, c2):
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


class Center:
    def __init__(self, center=(0, 0), score=0.):
        self.c = center
        self.s = score

    def __str__(self):
        return f'position : {self.c}, score : {self.s}'

    def distTo(self, other):
        return dist(self.c, other.c)

    def average(self, other):
        c1 = np.array(self.c)
        c2 = np.array(other.c)
        self.c = tuple((self.s * c1 + other.s * c2) / (self.s + other.s))
        self.s = self.s + other.s

    def normalize_score(self, max, min):
        """to prevent negative scores"""
        self.s = (self.s - min)/(max-min)

    def nearest(self, centers):
        """returns the nearest center to self in the list of center and its idx in the list"""
        d_min = 10**10
        idx = 0
        if len (centers) == 0:
            return None, idx, d_min
        for i,c in enumerate(centers):
            d = self.distTo(c)
            if d < d_min:
                d_min = d
                idx = i
        return centers[idx], idx, d_min





def get_center_dict_from_txt(center_txt, threshold=-100, nb_center_per_slice=2, normalize=True):
    """Take as input a  txt file which contains centers of particules computed by topaz and returns a dictionnary,
    keys are names of images and values a list of centers. All the centers with a score less than threshold are not taken
    into account. It keeps for each images only nb_center_per_slice centrioles. (the ones with highest scores)

    If normalize = True : normalize scores between 0 and 1
     """
    dict = {}

    with open(center_txt) as f:
        lines = f.readlines()

    lines = [line[:-1].split('\t') for line in lines[1:]]
    scores = []
    for line in lines:

        im_name = f'{line[0]}.tiff'
        center = (int(line[2]), int(line[1])) #PIL convention != numpy convention
        value = float(line[3])

        if value >=threshold:
            scores.append(value)
            if dict.get(im_name) is None:
                dict[im_name] = [Center(center, value)]
            else:
                if not len(dict[im_name]) >= nb_center_per_slice:
                    dict[im_name].append(Center(center, value))
    if normalize:
        ma = max(scores)
        mi = min(scores)
        for im_name in dict.keys():
            for c in dict[im_name]:
                c.normalize_score(ma, mi)

    return dict


def train_topaz_model(set_topaz, model_name='centriole_detection', epochs=100, particle_size=60, stats_file='model_training.txt'):

    '''train a neural network which detects centrioles
    First it resclales images if needed so that centriole size match the receptieve field of the topaz model
    set_topaz : folder which contains 2 subfolder train and valid and 2 txt files with the labelled positions of centrioles,
    train_label.txt and valid_label.txt
    save the results in {models}/{model_name}'''
    # preprocess
    receptive_field = 71
    downsampling_scale = int(particle_size / receptive_field + 0.5)

    for split in ['train', 'valid']:
        output = f'{set_topaz}/{split}_processed'
        if not os.path.exists(output):
            os.makedirs(output)
        cmd = 'topaz preprocess'
        cmd += ' -v'
        cmd += f' -s {downsampling_scale}'
        cmd += f' -o {output}'
        cmd += f' {set_topaz}/{split}/*.tiff'
        if downsampling_scale != 1:
            os.system(cmd)

        cmd = 'topaz convert'
        cmd += f' -s {downsampling_scale}'
        cmd += f' -o {set_topaz}/{split}_labels_processed.txt'
        cmd += f' {set_topaz}/{split}_labels.txt'
        if downsampling_scale != 1:
            os.system(cmd)

    # this command takes the particle coordinates matched to the original micrographs
    # and scales them by 1/8 (-s is downscaling)
    # the -x option applies upscaling instead
    # topaz convert -s 8 -o data/EMPIAR-10025/processed/particles.txt data/EMPIAR-10025/rawdata/particles.txt
    train = 'train' if downsampling_scale == 1 else 'train_processed'
    valid = 'valid' if downsampling_scale == 1 else 'valid_processed'
    train_labels = 'train_labels' if downsampling_scale == 1 else 'train_labels_processed'
    valid_labels = 'valid_labels' if downsampling_scale == 1 else 'valid_labels_processed'
    print('train', train)
    # train
    cmd = 'topaz train'
    cmd += ' -n 2' #number of particles per micrograph
    cmd += f' --train-images {set_topaz}/{train}'
    cmd += f' --train-targets {set_topaz}/{train_labels}.txt'
    cmd += f' --test-images {set_topaz}/{valid}'
    cmd += f' --test-targets {set_topaz}/{valid_labels}.txt'
    cmd += f' --save-prefix {pth.models}/{model_name}'
    cmd += f' -o {pth.myHome}/{stats_file}.txt'
    cmd += ' --radius 5'
    cmd += ' --epoch-size 40'
    cmd += f' --num-epochs {epochs}'
    os.system(cmd)


if __name__ == '__main__':

    cell_images = f'{pth.myHome}/wide_field/wide_field_cell_images'
    compute_center_particles_using_trained_neunet_model(cell_images, relative='deconv/c2',
                                                        model_name='centriole_detection_epoch100.sav', centriole_size=20, tile_sz=512)
    #d = get_center_dict_from_txt(f'{pth.myHome}/center_particles_test_cell.txt', threshold=0)
    #print(d)
    #train_topaz_model(f'{pth.training_sets}/particle_detection')
    #compute_center_particles_using_trained_neunet_model(f'{pth.myHome}/wide_field/wide_field_resized', relative='deconv/c2', centriole_size=30,
                                                        #model_name='centriole_detection_epoch100.sav')