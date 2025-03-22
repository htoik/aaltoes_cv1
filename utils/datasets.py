import os
from PIL import Image
from torch.utils.data import Dataset

class AaltoesDataset(Dataset):
    def __init__(self, root_dir, mode="train", transform=None):
        self.mode = mode
        self.transform = transform

        if mode == "train":
            self.image_dir = os.path.join(root_dir, "train", "train", "images")
            self.mask_dir = os.path.join(root_dir, "train", "train", "masks")
            self.orig_dir = os.path.join(root_dir, "train", "train", "originals")
            self.image_files = sorted(os.listdir(self.image_dir))
            mask_files = sorted(os.listdir(self.image_dir))
            orig_files = sorted(os.listdir(self.image_dir))
            self.max_idx = min([len(self.image_files), len(mask_files), len(orig_files)])
        elif mode == "test":
            self.image_dir = os.path.join(root_dir, "test", "test", "images")
            self.image_files = sorted(os.listdir(self.image_dir))
            self.max_idx = len(self.image_files)
        else:
            raise ValueError("Mode must be 'train' or 'test'")

    def __len__(self):
        # return len(self.image_files)
        return self.max_idx

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_name)

        image = Image.open(image_path).convert("RGB")

        if self.mode == "train":
            mask_path = os.path.join(self.mask_dir, img_name)
            orig_path = os.path.join(self.orig_dir, img_name)

            mask = Image.open(mask_path).convert("L")
            try:
                original = Image.open(orig_path).convert("RGB")
            except FileNotFoundError:
                original = None

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
                if original is not None:
                    original = self.transform(original)

            sample = {"image": image, "mask": mask}
            # sample = {"image": image, "mask": mask, "original": original}
        else:
            if self.transform:
                image = self.transform(image)
            sample = {"image": image}

        return sample
