import sys
sys.path.insert(0, './ext/HiFi_IFDL')
from HiFi_Net import HiFi_Net, get_config
from PIL import Image
import numpy as np
import os

get_config('./config/config.yml')
# path_to_data = "ext/HiFi_IFDL/asset"
path_to_data = "data/test1"

img_paths = [os.path.join(path_to_data, x) for x in os.listdir(path_to_data)]
# img_paths = [
#     # os.path.join(path_to_data, "sample_1.jpg"),
#     # os.path.join(path_to_data, "sample_3.jpg"),
#     # os.path.join(path_to_data, "sample_4.png"),
#     os.path.join(path_to_data, "image_46.png"),
# ]
for i,img_path in enumerate(sorted(img_paths)):
    HiFi = HiFi_Net()   # initialize
    ## detection
    res3, prob3 = HiFi.detect(img_path)
    # print(res3, prob3) 1 1.0
    HiFi.detect(img_path, verbose=True)

    ## localization
    binary_mask = HiFi.localize(img_path)
    binary_mask2 = np.asarray(binary_mask)
    print(binary_mask2.max())
    print(binary_mask2.min())

    binary_mask = Image.fromarray((binary_mask*255.).astype(np.uint8))
    binary_mask.save(f'output_image_test_{i:04d}.png')
