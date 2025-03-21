from HiFi_Net import HiFi_Net 
from PIL import Image
import numpy as np

HiFi = HiFi_Net()
img_path = 'asset/sample_1.jpg'

res3, prob3 = HiFi.detect(img_path)
HiFi.detect(img_path, verbose=True)

binary_mask = HiFi.localize(img_path)
binary_mask = Image.fromarray((binary_mask*255.).astype(np.uint8))
binary_mask.save('pred_mask.png')