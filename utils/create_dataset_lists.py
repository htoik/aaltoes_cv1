import os
from utils import create_dataset_list

def main():
    split = 'train'
    name = 'kaggle_aaltoes_cv1_small'
    src = os.path.expanduser(f'~/workspace/aaltoes_cv1/{name}/{split}')
    dst = f'data/dataset_splits'

    create_dataset_list(src, dst, name, 'images', 'masks')
    # create_dataset_list(src, dst, f"{name}_{split}", 'images', 'masks', 'originals')

if __name__ == '__main__':
    main()
