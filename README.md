# 📊 EDA Visualizations for Image Recognition (Conv Filter Edition)

## Overview

This project aims to provide insightful exploratory data analysis (EDA) for image recognition tasks, specifically focusing on visualizing convolutional filters in deep learning models. With visualizations of feature maps and convolution layers, we can gain a better understanding of how these models process images.

---

## 🛠️ Dependencies

To run this project, you'll need the following Python packages:

- **PyTorch**: Version 1.10.0
- **Torchvision**: Version 0.11.1
- **Torchaudio**: Version 0.10.0
- **Timm**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **OpenCV**
- **Pillow**

You can install the required packages using:

```bash
!pip install -q timm
!pip install -q torch==1.10.0 torchvision==0.11.1 torchaudio===0.10.0
```

---

## 📚 Config and Logging

A logging mechanism is implemented to facilitate debugging and track the flow of the program. Here's how to initialize the logger:

```python
def init_logger(log_file: str = "info.log") -> logging.Logger:
    ...
```

---

## 🎨 Visualization Utilities

Several helper functions are provided for visualizing images and feature maps:

- **`plot_multiple_img`**: Displays multiple images in a grid format.
- **`conv_horizontal`**: Visualizes the horizontal convolution of an image.
- **`conv_vertical`**: Visualizes the vertical convolution of an image.

### Example of Plotting Images

```python
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
```

---

## 🔍 Convolution Layers

### Understanding Convolution

Convolution involves a kernel (a 2D matrix) that moves over the entire image, calculating dot products with each window. Below is a GIF illustrating convolution in action:

![Convolution GIF](https://i.imgur.com/wYUaqR3.gif)

---

## 🖼️ Feature Map Visualization

### Step 1: Initialize the Models

In this project, we leverage **PyTorch's** `feature_extraction` module to visualize feature maps from various pretrained models:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16_pretrained = models.vgg16(pretrained=True).to(device)
resnet18_pretrained = models.resnet18(pretrained=True).to(device)
```

### Step 2: Transform the Tensors

We prepare the input images by resizing and normalizing them:

```python
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Step 3: Plotting the Feature Maps

We visualize the feature maps using a custom plotting function:

```python
def plot_feature_maps(processed_feature_maps, feature_map_names, nrows, title=None):
    ...
```

### Example Output

- **VGG16 Pretrained Feature Maps**
- **ResNet18 Pretrained Feature Maps**

---

## ⏳ Comparison with Randomly Initialized Weights

We also visualize the differences in feature maps when using pretrained weights versus randomly initialized weights.

```python
_, vgg16_processed_feature_maps, vgg16_feature_map_names = get_feature_maps(
    model_name="vgg16", image=cat_tensor, reduction="mean", pretrained=False
)
```

### Example Outputs

- **VGG16 NOT Pretrained Feature Maps**
- **ResNet18 NOT Pretrained Feature Maps**

---

## 📖 References

- [PyTorch Feature Extraction Documentation](https://pytorch.org/vision/stable/feature_extraction.html)
- [Visualizing Feature Maps Using PyTorch](https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch)
- [PyTorch Blog on FX Feature Extraction](https://pytorch.org/blog/FX-feature-extraction-torchvision/)
- [Plant Pathology EDA Models on Kaggle](https://www.kaggle.com/tarunpaparaju/plant-pathology-2020-eda-models)

---

## 🤝 Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Customization Tips:
- Update paths and references to match your project.
- You may want to include additional sections like "Installation Instructions" or "Usage Examples" based on your audience's needs.
- Add any relevant badges (e.g., build status, license) at the top for better visibility.
