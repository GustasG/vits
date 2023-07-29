# vits-lithuanian

VITS TTS model fitted for lithuanian language.

## Installing

This project uses [conda](https://conda.io) for virtual environment and package management.

### Creating conda environment

1. ```conda env create -f environment.yml```
2. ```conda activate vits```

## Training

**Training required GPU with CUDA support!**

Download the training dataset, then rename or create a link to the dataset folder: ```ln -s files/datasets/atrinkti_sak_3/wav DUMMY1```

Run the following command to start training:

```bash
python train.py -c files/configs/lt.json -m lithuanian
```

## Dataset

Dataset is not added in this repository since it is quite big and github charges for [LFS](https://git-lfs.com/) usage if reached certain size treshold

## Inference

To test trained model you can use inference.ipynb notebook in notebooks directory. To use trained model parameters replace load_checkpoint with trained model path from logs directory
