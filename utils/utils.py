import cv2
import os
import shutil
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
import torchmetrics

def create_validation_split(idxs, p=0.2):
    l = len(idxs)
    val_l = int(l * p)
    val_idxs = random.sample(idxs, val_l)
    train_idxs = [i for i in idxs if i not in val_idxs]
    return train_idxs, val_idxs

def create_smaller_dataset(src_path, dst_path, *args, size=100, id_func=lambda x: int(x.split('_')[1].split('.')[0]), excluded_ids=[]):
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    args = list(args)
    ids = [id_func(x) for x in os.listdir(os.path.join(src_path, args[0]))]
    ids = [i for i in ids if i not in excluded_ids]
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
    return selected_ids

def create_dataset_list(src_path, dst_fp, name, *args):
    if not os.path.exists(dst_fp):
        os.mkdir(dst_fp)
    files_list = []
    for a in args:
        src_fp = os.path.join(src_path, a)
        files = []
        for fn in os.listdir(src_fp):
            files.append(os.path.join(src_fp, fn))
        files_list.append(files)
    dst_file = os.path.join(dst_fp, f"{name}_{a}_list.txt")
    with open(dst_file, 'w') as f:
        csv_rows = [','.join(files) for files in zip(*files_list)]
        f.write('\n'.join(csv_rows))
        f.write('\n')

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

def calculate_iou_jaccard(model, data_loader, device):
    jaccard_index = torchmetrics.JaccardIndex(task='binary',num_classes=1)
    jaccard_index = jaccard_index.to(device)
    with torch.no_grad():
        for i,(images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = (labels[:,0,:,:]).unsqueeze(1).to(torch.int64)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1).unsqueeze(1)
            jaccard_index.update(preds, labels)
    avg_jaccard = jaccard_index.compute()
    return avg_jaccard
