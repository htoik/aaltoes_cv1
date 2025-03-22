import os
from PIL import Image
from torch.utils.data import Dataset

class AaltoesDataset(Dataset):
    def __init__(self, root_dir, mode="train", transform=None):
        self.mode = mode
        self.transform = transform
        self.include_full_forged = True
        self.dataset = {}

        if mode == "train":
            self.image_dir = os.path.join(root_dir, "train", "train", "images")
            self.mask_dir = os.path.join(root_dir, "train", "train", "masks")
            self.orig_dir = os.path.join(root_dir, "train", "train", "originals")

            dict1 = [(int(img.split('_')[1].split('.')[0]), img) for img in self.image_dir]
            dict2 = [(int(img.split('_')[1].split('.')[0]), img) for img in self.mask_dir]
            dict3 = [(int(img.split('_')[1].split('.')[0]), img) for img in self.orig_dir]

            ks = set(dict1.keys()).add(set(dict2.keys(()))).add(set(dict3.keys(())))
            self.img_idxs = list(range(len(ks)))
            self.img_to_idxs = dict(zip(sorted(ks), self.img_idxs))

            self.dataset = dict((self.img_to_idxs[k], (None, None, None)) for k in ks)
            for idx, img in dict1: self.dataset[self.img_to_idxs[idx]][0] = img
            for idx, img in dict2: self.dataset[self.img_to_idxs[idx]][1] = img
            for idx, img in dict3: self.dataset[self.img_to_idxs[idx]][2] = img
        elif mode == "test":
            self.image_dir = os.path.join(root_dir, "test", "test", "images")
            # self.image_files = sorted(os.listdir(self.image_dir))
            dict1 = [(int(img.split('_')[1].split('.')[0]), img) for img in self.image_dir]
            ks = set(dict1.keys()).add(set(dict2.keys(()))).add(set(dict3.keys(())))
            self.img_idxs = list(range(len(ks)))
            self.img_to_idxs = dict(zip(sorted(ks), self.img_idxs))

            self.dataset = dict((self.img_to_idxs[k], (None, None, None)) for k in ks)
            for idx, img in dict1: self.dataset[self.img_to_idxs[idx]][0] = img
        else:
            raise ValueError("Mode must be 'train' or 'test'")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name, mask_name, orig_name = self.dataset[idx]
        image_path = os.path.join(self.image_dir, img_name)

        image = Image.open(image_path).convert("RGB")

        if self.mode == "train":
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = Image.open(mask_path).convert("L")

            try:
                orig_path = os.path.join(self.orig_dir, orig_name)
                original = Image.open(orig_path).convert("RGB")
            except FileNotFoundError:
                original = Image.new('RGB', image.size[1::-1])

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
                original = self.transform(original)

            sample = {"image": image, "mask": mask, "original": original}
        else:
            if self.transform:
                image = self.transform(image)
            sample = {"image": image}

        return sample
