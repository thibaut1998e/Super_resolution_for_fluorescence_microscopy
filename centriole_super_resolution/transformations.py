import imageio
import numpy as np
import os
import random as rd

from skimage import io
from skimage import feature as ft
import scipy.ndimage as scp
import imageio
import time as t
import copy as cp
from skimage.feature.blob import blob_dog, blob_doh, blob_log
import plot_graphs as pg
from functools import partial
#from inference import prediction


"""
Some transformation function to apply to images, input are 2d or 3d arrays, output 2D or 3D array or lists of 2D or 3D array
These functions can then be used with the method apply_transformation
"""


def resize(in_array, **kwargs):
    """resize in_array with upsampling scale given as argument
    if in_array is a 3D array it doesnt resize the third dimension"""
    if len(in_array.shape) == 2:
        order = kwargs.get('order')
        if order is None:
            order = 3
        scale = kwargs.get("scale")
        resized = scp.zoom(in_array, scale, order=order)
        return resized.astype(np.float32)
    else:
        return np.array([resize(in_array[i], **kwargs) for i in range(len(in_array))])


def convolution(in_array, sigma, **kwargs):
    convolved_image = scp.gaussian_filter(in_array, [sigma, sigma])
    return convolved_image


def random_crop2D(in_array, x_size, y_size=None,**kwargs):
    """crop randomly a tile of x_size*y_size"""
    if y_size is None: y_size = x_size
    x, y = in_array.shape
    rdx = rd.randint(0, x-x_size-1)
    rdy = rd.randint(0, y-y_size-1)
    return in_array[rdx:rdx+x_size, rdy:rdy+y_size]


def crop(array, min_x, max_x, min_y, max_y, **kwargs):
    """crop a tile delimited by x_min, y_min, x_max, y_max"""
    h, w = array.shape
    min_x, max_x, min_y, max_y = int(max(min_x, 0)), int(min(max_x, h)), int(max(min_y, 0)), int(min(max_y, w))
    cropped = array[min_x:max_x, min_y:max_y]
    return cropped


def crop_center(in_array, **kwargs):
    """crop a tile of size x_size*y_size in the center of the image. """
    if len(in_array.shape) == 2:
        x_size = kwargs.get('x_size')
        y_size = kwargs.get('y_size')
        if y_size is None: y_size = x_size
        midx, midy = len(in_array) // 2, len(in_array[0]) // 2
        min_x, min_y = midx - x_size // 2, midy - y_size // 2
        max_x, max_y = min_x + x_size, min_y + y_size
        in_array_cropped = crop(in_array, min_x=min_x, max_x=max_x, min_y=min_y, max_y = max_y)
        return in_array_cropped
    #3D array
    else:
        return np.array([crop_center(in_array[i], **kwargs) for i in range(len(in_array))])


def cross_section(array3D, first=10, last=None, **kwargs):
    """returns a section of the image of maximal intensity. It doesnt take into account the 10 first slices because there
    is often noise in these images"""
    if last is None: last = array3D.shape[0]
    max_intensity = 0
    imax = 0
    for i in range(first, last):
        intensity = np.sum(array3D[i])
        if intensity >= max_intensity:
            imax = i
            max_intensity = intensity
    return array3D[imax]


def cross_section_slices(array3D, first=10, last=None, p=1/4, **kwargs):
    """returns a list of slices between first and last which have an Intensity more than I_max*p"""

    if last is None: last = array3D.shape[0]
    max_intensity = max([np.sum(array3D[i]) for i in range(len(array3D))])
    slices = []
    for i in range(first, last):
        intensity = np.sum(array3D[i])
        if intensity >= p*max_intensity:
            slices.append(array3D[i])
    return slices


def pad_center(array, x_size, y_size, **kwargs):
    """pad the input array with 0 such that the output array has a shape (x_size, y_size)"""
    x, y = array.shape
    if x > x_size or y > y_size:
        raise("you can not padd to get an array of smaller size than the input")
    x_pad, y_pad = (x_size-x)/2, (y_size-y)/2

    def get_before_after_pad(pad):
        if int(pad) == pad:
            pad_before = pad
            pad_after = pad
        else:
            pad_before = int(pad)
            pad_after = int(pad) + 1
        return int(pad_before), int(pad_after)
    x_pad_before, x_pad_after = get_before_after_pad(x_pad)
    y_pad_before, y_pad_after = get_before_after_pad(y_pad)
    #print(((x_pad_before, x_pad_after), (y_pad_before, y_pad_after)))
    padded_array = pad(array, xb=x_pad_before, xa=x_pad_after, yb=y_pad_before, ya=y_pad_after)
    return padded_array

def pad(array, xb, yb, xa, ya, **kwargs):
    xb, yb, xa, ya = int(xb), int(yb), int(xa), int(ya)
    padded = np.pad(array, ((xb, xa), (yb, ya)), mode='constant')
    return padded


def crop_or_padd_center(array, x_size, y_size, **kwargs):
    """return an output array of shape (x_size, y_size), if the input array is smaller it padds with 0,
    otherwise it cropps the image"""
    x_size, y_size = int(x_size), int(y_size)
    x, y = array.shape
    if x <= x_size and y <= y_size:
        return pad_center(array, x_size=x_size, y_size=y_size)
    elif x > x_size and y > y_size:
        return crop_center(array, x_size=x_size, y_size=y_size)

    else:
        return None


def divide_by_max(array, **kwargs):
    return (array/np.amax(array)).astype(np.float32)


def normalize(array, **kwargs):
    mean, std = np.mean(array), np.std(array)
    return ((array-mean)/std).astype(np.float32)


def norm_min_max(array, **kwargs):
    mi, ma = np.min(array), np.max(array)
    return ((array-mi)/(ma-mi)).astype(np.float32)


def add_guassian_spot(array, **kwargs):
    """add gaussians spots to mimic the background of wide field images.
    freq : 1 pixel over 1/freq is center of a spot
    ampl : [mean_ampl, std_ampl]
    sigma_s : [mean_sigma, std_sigma]
    spot_width : spots are cut after this radius so that the computation is not to long"""

    mean_sigma, std_sigma = kwargs.get('sigma_s')
    mean_ampl, std_ampl = kwargs.get('ampl')
    freq = kwargs.get('freq')
    spot_width = kwargs.get('spot_width')
    half = spot_width//2
    h, w = array.shape
    nb_spots = int(freq*h*w)
    out_array = cp.deepcopy(array)
    max_intensity = np.amax(array)
    for i in range(nb_spots):
        spot = np.random.randint(0, min(h,w), 2)
        sigma = np.random.normal(mean_sigma, std_sigma)
        sigma = max(sigma, 0.1)
        ampl = np.random.normal(mean_ampl, std_ampl)
        for x in range(spot[0]-half, spot[0]+half):
            for y in range(spot[1]-half, spot[1]+half):
                if x >= 0 and x < out_array.shape[0] and y >=0 and y < out_array.shape[1]:
                    dist = np.linalg.norm(spot-np.array([x,y]))
                    noise = gauss(dist, sigma, max_intensity*ampl)
                    out_array[x][y] += noise

    return out_array.astype(np.float32)


def gauss(x, sigma, ampl):
    return ampl*np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))


def crop_with_center(array, **kwargs):
    """crop a tile of tile size centred in center. Then pad if needed and if use_pad = True"""
    if len(array.shape) == 2:
        center = kwargs.get('center')
        tile_sz = kwargs.get('tile_sz')
        use_pad = kwargs.get('pad')
        if use_pad is None: use_pad = True
        xc, yc = center
        h, w = array.shape

        cropped = crop(array, min_x=xc-tile_sz//2, max_x=xc+tile_sz//2, min_y=yc-tile_sz//2, max_y=yc+tile_sz//2)

        if use_pad:
            pad_x_before = max(tile_sz // 2 - xc, 0)
            pad_y_before = max(tile_sz // 2 - yc, 0)
            pad_x_after = max(xc + tile_sz // 2 - h, 0)
            pad_y_after = max(yc + tile_sz // 2 - h, 0)
            cropped = pad(cropped, xb=pad_x_before, xa=pad_x_after, yb=pad_y_before, ya=pad_y_after)

        return cropped
    else:
        return np.array([crop_with_center(array[i], **kwargs) for i in range(len(array))])


def add_noise(array, sigma_noise, **kwargs):
    """add gaussian noise of standard deviation sigma"""
    h, w = array.shape
    array_out = np.zeros((h,w))
    max = np.max(array)
    for i in range(h):
        for j in range(w):
            array_out[i][j] = array[i][j] + np.random.normal(0, sigma_noise*max)
    return array_out.astype(np.float32)


def crop_with_edges_info(array, low, high, marge, sigma, tile_sz, **kwargs):
    """crop the array using an edge detector algorithm to detect particles"""
    edges = compute_edge_array(array, low, high, sigma)
    x_min, x_max, y_min, y_max = find_limits(edges)

    if not x_min:
        print('crop center')
        cropped = crop_center(array, x_size=tile_sz, y_size=tile_sz)

    else:
        if tile_sz:
            center = [(x_min+x_max)//2, (y_min+y_max)//2]
            cropped = crop_with_center(array, center=center, tile_sz=tile_sz)
        else:
            cropped = crop(array, min_x=x_min-marge, min_y=y_min-marge, max_x=x_max+marge, max_y=y_max+marge)

    return cropped


def find_limits(edge_array):
    """given a boolean edge array compute the corners of the tile containing (ideally) the centriole"""
    h, w = edge_array.shape
    x_min = 0
    while x_min < h and not edge_array[x_min].any():
        x_min += 1
    if x_min == h:
        return None, None, None, None
    x_max = h-1
    while not edge_array[x_max].any():
        x_max -= 1
    y_min = 0
    while not edge_array[:, y_min].any():
        y_min += 1
    y_max = w-1
    while not edge_array[:, y_max].any():
        y_max -= 1
    return x_min, x_max, y_min, y_max

def compute_edge_array(array, low=0.3, high=0.9, sigma=5, ma=None):
    if not ma:
        ma = np.amax(array)
    out_image = ft.canny(array, sigma=sigma, low_threshold=low*ma, high_threshold=high*ma)
    return out_image


def transform_bool_in_binary(array):
    return np.array(array, dtype=np.int16)


def binary_edge_array(array, sigma, low, high, ma, **kwargs):
    bool_edge_array = compute_edge_array(array, low, high, sigma, ma)
    binary_edge_array = transform_bool_in_binary(bool_edge_array)
    return binary_edge_array


def crop_with_center_dict(array, image_name, center_dict, tile_sz, **kwargs):
    """crop with center_dict computed with topaz"""
    centers = center_dict.get(image_name)
    tiles = []
    if centers is None or len(centers) == 0:
        print(f' WARNING : no center for image {image_name}')
    else:
        for center in centers:
            tiles.append(crop_with_center(array, center=center.c, tile_sz=tile_sz))
    return tiles


def crop_with_center_dict_2(array, image_name, center_dict, tile_sz):
    """same as above but instead of returning a list of tiles returns a dictionnary, keys are centers and values tiles"""
    centers = center_dict.get(image_name)

    tiles = {}
    if centers is None: centers = []
    if len(centers) == 0:
        print(f' WARNING : no center for image {image_name}')
    else:
        for center in centers:
            tiles[center.c] = crop_with_center(array, center=center.c, tile_sz=tile_sz)
    return tiles


def crop_threeD_with_center_dict(array_3D, image_name, center_dict, tile_sz=50, radius=30, nb_of_crop=None, cut=False,
                                 average=True, save_figure=False, **kwargs):
    """
    a 3D detection of centrioles using topaz results for all the slices. Returns a list of 3D arrays

    image_name : str, name of the image (with extension)

    center_dict : dict got from method 'get_center_dict_from_txt' which contains the positions of centers for all the slices

    tile_sz : int, optional size of patches

    cut : boolean optional, if true cut the image at local mimima of intensity

    n : nb_of_crop, int optiopnal : number of patches to crop in the image (with highest scores).

    average, boolean optional. If True, center of one centriole in the 3D image is computed as
    the average over the slices of centers closer than radius. If False, it is computed as the center of the slice of
    highest intensity

    radius, int optional. Center closer than radius (in 2 different slices or in the same slice) are considered represent
    the same particle

    save_figure : optional, default False : save figures of intensity graphs

      """

    image_name = image_name.split('.')[0]

    if len(array_3D.shape) != 3:
        print(f'{image_name} not a 3D array, nb of dimension : {len(array_3D.shape)}')
        raise ValueError

    average_position_centers = []
    for i in range(len(array_3D)):
        key = f'{image_name}_slice_{i}.tiff'
        if center_dict.get(key) is not None:
            centers = center_dict[key]
            for c in centers:
                _, nearest, min_dist = c.nearest(average_position_centers)
                if min_dist < radius:
                    #new_average = (np.array(c) + cpts[nearest]*average_position_centers[nearest])/(cpts[nearest]+1)
                    average_position_centers[nearest].average(c)

                else:
                    average_position_centers.append(c)

    average_position_centers.sort(key=lambda x: x.s, reverse=True)
    if nb_of_crop is not None:
        average_position_centers = average_position_centers[:nb_of_crop]
    _, h, w = array_3D.shape
    centrioles = []
    for i,c in enumerate(average_position_centers):
        threeD_tile = crop_with_center(array_3D, tile_sz=tile_sz, center=c.c)
        min_locs, max_locs, argmin, argmax = pg.intenity_graph(threeD_tile, f'{image_name}_{i}', save_figure=save_figure)
        if len(min_locs) == 0 or len(max_locs) == 0 or max_locs[0] < min_locs[0]:
            min_locs.insert(0, 0)
        if not average:
            if not cut:
                centers_slice_highest_intensity = get_slice_centers(center_dict, image_name, argmax, c)
                center, _, _ = c.nearest(centers_slice_highest_intensity)
                centrioles.append(crop_with_center(array_3D, tile_sz=tile_sz, center=center.c))
            else:
                for i in range(len(min_locs)-1):
                    centers_slice_highest_intensity = get_slice_centers(center_dict, image_name, max_locs[i], c)
                    center, _ = c.nearest(centers_slice_highest_intensity)
                    threeD_tile = crop_with_center(array_3D, tile_sz=tile_sz, center=center.c)[min_locs[i]:min_locs[i+1]+1, :, :]
                    centrioles.append(threeD_tile)


        else:
            if cut:
                centrioles += [threeD_tile[min_locs[i]:min_locs[i + 1] + 1, :, :] for i in range(len(min_locs) - 1)]
            else:
                centrioles.append(threeD_tile)

    return centrioles


def get_slice_centers(center_dict, im_name, slice_id, center):
    """return the centers of slice slice_id, if it is not in the dictionnary it returns the nearest slice
    which is in the dictionnary and print a warning message"""
    if center_dict.get(f'{im_name}_slice_{slice_id}.tiff') is not None:
        centers = center_dict[f'{im_name}_slice_{slice_id}.tiff']
    else:
        old = slice_id
        i = 0
        while center_dict.get(f'{im_name}_slice_{slice_id}.tiff') is None:
            slice_id = slice_id + (-1) ** i * (i + 1)
            i += 1
        centers = center_dict[f'{im_name}_slice_{slice_id}.tiff']
        print(
            f'WARNING : no center for slice {old} (slice of highest intensity or local maximum of intensity'
            f') in image {im_name}, average center {center.c}, '
            f'image was cropped instead at the center predicted for slice {slice_id}')
    return centers









if __name__ == "__main__":
    a = 1








