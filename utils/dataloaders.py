import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import AaltoesDataset

def get_aaltoes_dataloaders():
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    train_dataset = AaltoesDataset(root_dir="data", mode="train", transform=transform)
    test_dataset = AaltoesDataset(root_dir="data", mode="test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    return train_loader, test_loader