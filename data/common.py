﻿import os
import random
import numpy as np
import scipy.misc as misc
import imageio
from tqdm import tqdm

import torch

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
BINARY_EXTENSIONS = ['.npy']
BENCHMARK = ['IK', 'WV2', 'P', 'SP', 'QB']


####################
# Files & IO
####################
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_binary_file(filename):
    return any(filename.endswith(extension) for extension in BINARY_EXTENSIONS)


def _get_paths_from_images(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '[%s] has no valid image file' % path
    return images


def _get_paths_from_binary(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    files = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_binary_file(fname):
                binary_path = os.path.join(dirpath, fname)
                files.append(binary_path)
    assert files, '[%s] has no valid binary file' % path
    return files


def get_image_paths(data_type, dataroot, subset=None):
    paths = None
    if dataroot is not None:
        if data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        elif data_type == 'npy':
            if dataroot.find('_npy') < 0 :
                old_dir = dataroot
                dataroot = dataroot + '_npy'
                if not os.path.exists(dataroot):
                    print('===> Creating binary files in [%s]' % dataroot)
                    os.makedirs(dataroot)
                    img_paths = sorted(_get_paths_from_images(old_dir))
                    path_bar = tqdm(img_paths)
                    for v in path_bar:
                        img = imageio.imread(v, pilmode='RGB')
                        ext = os.path.splitext(os.path.basename(v))[-1]
                        name_sep = os.path.basename(v.replace(ext, '.npy'))
                        np.save(os.path.join(dataroot, name_sep), img)
                else:
                    print('===> Binary files already exists in [%s]. Skip binary files generation.' % dataroot)

            paths = sorted(_get_paths_from_binary(dataroot))

        else:
            raise NotImplementedError("[Error] Data_type [%s] is not recognized." % data_type)
    if subset is None:
        return paths
    start = int(subset[0]*len(paths))
    end = -1 if subset[1] == 1 else int(subset[1]*len(paths))
    return paths[start:end]


def find_benchmark(dataroot):
    bm_list = [dataroot.find(bm)>=0 for bm in BENCHMARK]
    if not sum(bm_list) == 0:
        bm_idx = bm_list.index(True)
        bm_name = BENCHMARK[bm_idx]
    else:
        bm_name = 'MyImage'
    return bm_name


def read_img(path, data_type):
    # read image by misc or from .npy
    # return: Numpy float32, HWC, RGB, [0,255]
    if data_type == 'img':
        img = imageio.imread(path, pilmode='RGB')
    elif data_type.find('npy') >= 0:
        img = np.load(path).astype('float32')
    else:
        raise NotImplementedError

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


####################
# image processing
# process on numpy image
####################

def np2Tensor(l, run_range, img_range):
    def _np2Tensor(img):
        # if img.shape[2] == 3: # for opencv imread
        #     img = img[:, :, [2, 1, 0]]
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(run_range / img_range)

        return tensor

    return [_np2Tensor(_l) for _l in l]


def pan_Tensor2np(tensor_list, run_range, img_range):

    def _Tensor2numpy(tensor, run_range):
        array = np.transpose(quantize(tensor, run_range, img_range).numpy(), (1, 2, 0)).astype(np.uint16)
        return array

    return [_Tensor2numpy(tensor, run_range) for tensor in tensor_list]


def quantize(img, rgb_range, img_range):
    pixel_range = img_range / rgb_range
    # return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    return img.mul(pixel_range).clamp(0, int(img_range)).round()


def get_patch(img_in, img_tar, img_pan, patch_size, scale, lrpan=None, msx2=False):
    ih, iw = img_in.shape[:2]
    oh, ow = img_tar.shape[:2]

    ip = patch_size

    if ih == oh:
        tp = ip
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = ix, iy
    else:
        tp = ip * scale
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy
        if msx2:
            msp = ip*2
            msx, msy = 2*ix, 2*iy


    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]
    img_pan = img_pan[ty:ty + tp, tx:tx + tp, :]
    if lrpan is not None:
        if msx2:
            lrpan = lrpan[msy:msy + msp, msx:msx + msp, :]
        else:
            lrpan = lrpan[iy:iy + ip, ix:ix + ip, :]

        return img_in, img_tar, lrpan, img_pan
    else:
        return img_in, img_tar, img_pan


def add_noise(x, noise='.'):
    if noise is not '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def modcrop(img_in, scale):
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [%d].' % img.ndim)
    return img
