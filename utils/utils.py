import cv2
import os
import shutil
import numpy as np
import pandas as pd
import random

def create_validation_split(idxs, p=0.2):
    l = len(idxs)
    val_l = int(l * p)
    val_idxs = random.sample(idxs, val_l)
    train_idxs = [i for i in idxs if i not in val_idxs]
    return train_idxs, val_idxs

def create_smaller_dataset(src_path, dst_path, *args, size=100, id_func=lambda x: int(x.split('_')[1].split('.')[0])):
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    args = list(args)
    ids = [id_func(x) for x in os.listdir(os.path.join(src_path, args[0]))]
    selected_ids = random.sample(ids, size)
    for a in args:
        srcp = os.path.join(src_path, a)
        dstp = os.path.join(dst_path, a)
        if not os.path.exists(dstp):
            os.makedirs(dstp, exist_ok=True)
        for f in os.listdir(srcp):
            if id_func(f) not in selected_ids:
                continue
            sfp = os.path.join(srcp, f)
            dfp = os.path.join(dstp, f)
            shutil.copyfile(sfp, dfp)

# From: https://www.kaggle.com/competitions/aaltoes-2025-computer-vision-v-1/data
def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# From: https://www.kaggle.com/competitions/aaltoes-2025-computer-vision-v-1/data
def rle2mask(mask_rle: str, label=1, shape=(256, 256)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape) 
