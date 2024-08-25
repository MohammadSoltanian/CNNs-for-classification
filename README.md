markdown
# EDA Visualizations for Image Recognition (Conv Filter Edition)

Welcome to the EDA Visualizations for Image Recognition project! This repository contains code and resources for exploring and visualizing convolutional filters in image recognition models using PyTorch. 

## Table of Contents
- [Dependencies and Imports](#dependencies-and-imports)
- [Config and Logging](#config-and-logging)
- [Utils](#utils)
- [Seeding](#seeding)
- [Transforms Params](#transforms-params)
- [Visualizations](#visualizations)
- [Convolution Layers](#convolution-layers)
- [Feature Extractor using PyTorch's native Feature Extraction Module](#feature-extractor-using-pytorchs-native-feature-extraction-module)
- [Visualizing VGG16 and ResNet18](#visualizing-vgg16-and-resnet18)
- [Comparison with Randomly Initialized Weights](#comparison-with-randomly-initialized-weights)
- [References](#references)

## Dependencies and Imports
To get started, make sure you have the necessary dependencies installed. You can do this by running the following commands:

bash
pip install -q timm
pip install -q torch==1.10.0 torchvision==0.11.1 torchaudio===0.10.0


Then, import the required libraries in your Python environment:

python
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


## Config and Logging
This section initializes logging to keep track of the operations performed during the execution of the code.

python
import logging
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

def init_logger(log_file: str = "info.log") -> logging.Logger:
logger = getLogger(__name__)
logger.setLevel(INFO)
stream_handler = StreamHandler()
stream_handler.setFormatter(Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
file_handler = FileHandler(filename=log_file)
file_handler.setFormatter(Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
return logger

logger = init_logger()


## Utils
Utility functions to help with visualizations and data processing.

python
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
fig, myaxes = plt.subplots(figsize=(20, 15), nrows=ceil(len(img_matrix_list) / ncols), ncols=ncols, squeeze=False)
fig.suptitle(main_title, fontsize=30)
fig.subplots_adjust(wspace=0.3, hspace=0.3)
for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
myaxes[i // ncols][i % ncols].imshow(img)
myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
plt.show()


## Seeding
To ensure reproducibility, we can set a random seed for all random number generators.

python
def seed_all(seed: int = 1992) -> None:
print(f"Using Seed Number {seed}")
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


## Transforms Params
Define the parameters for image transformations.

python
mean: List[float] = [0.485, 0.456, 0.406]
std: List[float] = [0.229, 0.224, 0.225]
image_size: int = 224

transform = torchvision.transforms.Compose([
torchvision.transforms.Resize((image_size, image_size)),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize(mean=mean, std=std),
])


## Visualizations
Visualize images and their features.

python
cat_p = "../input/petfinder-pawpularity-score/train/0042bc5bada6d1cf8951f8f9f0d399fa.jpg"
dog_p = "../input/petfinder-pawpularity-score/train/86a71a412f662212fe8dcd40fdaee8e6.jpg"

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
cat = PIL.Image.open(cat_p)
plt.imshow(cat)
plt.title("Cat")
plt.subplot(1, 2, 2)
dog = PIL.Image.open(dog_p)
plt.imshow(dog)
plt.title("Dog")
plt.show()


## Convolution Layers
Explore convolution layers and their effects on images.

python
def conv_horizontal(image: np.ndarray) -> None:
# Function implementation...

def conv_vertical(image: np.ndarray) -> None:
# Function implementation...


## Feature Extractor using PyTorch's native Feature Extraction Module
Utilize PyTorch's feature extraction capabilities to visualize convolutional layers.

python
def get_conv_layers(model: torchvision.models) -> Dict[str, str]:
# Function implementation...

def get_feature_maps(model_name: str, image: torch.Tensor, reduction: str = "mean", pretrained: bool = True) -> Union[Dict[str, torch.Tensor], List[torch.Tensor], List[str]]:
# Function implementation...


## Visualizing VGG16 and ResNet18
Visualize feature maps from popular models like VGG16 and ResNet18.

python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torchvision.models as models

vgg16_pretrained_true = models.vgg16(pretrained=True).to(device)
resnet18_pretrained_true = models.resnet18(pretrained=True).to(device)


## Comparison with Randomly Initialized Weights
Compare the feature maps of pretrained models with those initialized randomly.

python
# Code to visualize and compare feature maps...


## References
- PyTorch Feature Extraction Documentation
- Visualizing Feature Maps using PyTorch
- FX Feature Extraction in Torchvision
- Kaggle Plant Pathology EDA Models

---

Feel free to explore the code and contribute to the project! Happy coding! 😊
