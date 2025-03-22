# Aaltoes CV1

Contains competition stuff from the Aaltoes CV1 competition. We joined for the burgers.

## Installation

```bash
git clone --recursive https://github.com/htoik/aaltoes_cv1.git
conda env create -f conda/environment.yml
. source.sh
pip install -e .

# at some point you do this too, good luck
#conda install pytorch torchvision cudatoolkit=12.6
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# just try various things until it works
```

## 

## Ideas

### Data generation

- Image inpainting generative models

https://paperswithcode.com/task/image-inpainting

https://medium.com/data-science-at-microsoft/introduction-to-image-inpainting-with-a-practical-example-from-the-e-commerce-industry-f81ae6635d5e



## Detecting AI generated images

- patch-based forensic detectors

- Xception vs ViT

- https://github.com/ISICV/ManTraNet
