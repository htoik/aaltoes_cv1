from PIL import Image
import numpy as np
from dataloaders import get_aaltoes_dataloaders

def main():
    train_loader, test_loader = get_aaltoes_dataloaders()

    for batch in train_loader:
        images = batch["image"]
        masks = batch["mask"]
        # originals = batch["original"]
        print("Batch images shape:", images.shape)
        print("Batch masks shape:", masks.shape)
        # print("Batch originals shape:", originals.shape)
        break

if __name__ == "__main__":
    main()
