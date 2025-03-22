import os
from utils import create_smaller_dataset

def main():
    split = 'train'
    src = os.path.expanduser('~/workspace/aaltoes_cv1/kaggle_aaltoes_cv1')
    dst = os.path.expanduser('~/workspace/aaltoes_cv1/kaggle_aaltoes_cv1_small')
    if not os.path.exists(dst):
        os.makedirs(dst)
    create_smaller_dataset(
        os.path.join(src, split),
        os.path.join(dst, split),
        "images",
        "masks",
        "originals",
        size=256,
        id_func=lambda x: int(x.split('_')[1].split('.')[0])
    )

if __name__ == '__main__':
    main()
