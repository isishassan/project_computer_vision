# Computer Vision – CIFAR10 Image Classification using CNN and Transfer Learning

## 1. Project Overview

This project explores image classification using deep learning models (custom CNNs and transfer learning architectures) on the CIFAR-10 dataset to compare performance and identify best modeling approaches.

Group Project with [Alexandre Andrade](https://github.com/alexandrade1978) and [Janete Carina Caetano Barbosa](https://github.com/janeteccbarbosa28-eng) in the Ironhack Data Science and Machine Learning Bootcamp.


## 2. Dataset Description
Source: [CIFAR-10 dataset via Keras](https://keras.io/api/datasets/cifar10/)

### Dataset Overview:
- 60,000 total images
- Image size: 32 × 32 RGB
- 10 object classes, 6,000 images per class 

### Project Data Split:
- Train: 40,000
- Validation: 10,000
- Test: 10,000 

## 3. Research Goal / Question
Main goal: Compare performance of custom CNN vs transfer learning models on small image datasets.

Secondary questions:
- How much do pre-trained models improve performance vs custom CNNs?
- Does data augmentation improve generalization?
- How does resizing images affect transfer learning performance?

## 4. Steps You Took (Summary)
#### 1. Data Cleaning & Preparation
- Normalized image pixel values
- One-hot encoded labels
- Created train / validation / test splits

#### 2. Modeling
- Custom CNN
- Multiple convolution blocks
- Dropout layers for regularization
- Early stopping + learning rate scheduling 

#### 3. Transfer Learning Models
- EfficientNetB0
- EfficientNetV2B1
- EfficientNetV2S (PyTorch implementation) 

#### 4. Data Augmentation
- Rotation
- Brightness / contrast / saturation / hue changes
- Random crop
- Random hole cutout
- Flipping 

#### 5. Training Techniques
- Learning rate scheduler
- Label smoothing
- Partial layer unfreezing in transfer models 

## 5. Main Findings

- Transfer learning significantly outperformed custom CNN models.
- Data augmentation improved validation performance and robustness.
- Best performance achieved using stacked ensemble of transfer models.
- Final stacked model achieved ~96.9% F1 score. 

## 6. How to Reproduce the Project (Optional – Draft)

#### Prerequisites
- Python 3.x
- TensorFlow / Keras
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

#### Running the Project
Install requirements
Run notebooks in /notebooks
Models saved in /models
Performance reports generated automatically

## 7. Next Steps / Ideas for Improvement

Potential improvements:
- Try larger image sizes or higher-resolution datasets
- Test additional transfer architectures (Vision Transformers, ConvNeXt)
- Perform hyperparameter optimization (Optuna, Keras Tuner)
- Deploy best model as API or web app
- Improve stacking ensemble methodology

## 8. Repo Structure
repo/
- models: Saved trained models
- notebooks: Jupyter notebooks for experimentation and training
- reports: Visual outputs, plots, evaluation summaries
- utils: Helper functions (training, evaluation, preprocessing)
- model_performance_report: DataFrames / CSV reports tracking model metrics
- requirements.txt: Python dependencies for project reproducibility


## 9. Additional

### Reuse in other projects

Option 1 (simple): copy the `utils/` folder into the new project.
Option 2 (recommended): package it as a small internal library and install with pip in your other projects.


### Run order
1. `notebooks/01_data_processing.ipynb`
2. `notebooks/02_cnn_model.ipynb`
3. `notebooks/03_cnn_model_tuned.ipynb`
4. `notebooks/04_transfer_learning.ipynb`
5. `notebooks/05_cnn_model_jax_flax.ipynb`
6. `notebooks/06_transfer_learning_EfficientNetV2B1.ipynb`
7. `notebooks/07_transfer_learning_EfficientNetV2S_PyTorch.ipynb`
8. stacked model - notebook to be added
