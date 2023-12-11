# import necessary libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
from torchvision import transforms
import random

import torch
import torch.nn.functional as F
import torchvision
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split

import os
import pandas as pd

import json
import numpy as np

from keras.applications.resnet50 import ResNet50

# load_images: input - file path
#              output - list of images
#              retrieves each image at given filepath, transforms it to tensor, converts it to RGB, 
#              resizes it to be compatabile with ResNet and appends it to a list of images for that folder
def load_images(folder_path):
    image_list = []
    max_files_open = 10
    transform = transforms.Compose([transforms.PILToTensor()])
    file_list = os.listdir(folder_path)
    for i in range(0, len(file_list), max_files_open):
        batch = file_list[i:i + max_files_open]
        for filename in batch:
            with open(os.path.join(folder_path, filename), 'r') as file:
                img = Image.open(os.path.join(folder_path, filename)).convert('RGB')
                img_resized = img.resize((224, 224))
                img_resized = np.array(img_resized)
                if img_resized.shape[2] == 3:
                    image_list.append(img_resized)
    return image_list

# get images from downloaded kaggle datasets
mal1_train = load_images(<path to dataset>)
norm1_train = load_images(<path to dataset>)

# combine images and labels (use a subset of the data for testing purposes)
norms = [0] * (455)
mals = [1] * (460)
data = np.stack(norm1_train + norm1_test + norm2 + mal1_train + mal1_test + mal2)
labels = np.concatenate((norms,mals))

# create tensor dataset
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
batched_ds = dataset.batch(32)
train_ds, val_ds = tf.keras.utils.split_dataset(
    batched_ds, left_size=0.7, right_size=0.3, shuffle=True, seed=None
)

# seperate into training and testing datasets
x_test = np.concatenate([x for x, y in val_ds], axis=0)
y_test = np.concatenate([y for x, y in val_ds], axis=0)
x_train = np.concatenate([x for x, y in train_ds], axis=0)
y_train = np.concatenate([y for x, y in train_ds], axis=0)

# find the best best performing l2_norm_clip
l2_histories = []
l2_norm_clip = [0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 1.9, 2, 2.5] # play around
for l2 in l2_norm_clip:
    print("l2_norm_clip: ", l2)
    resnet_50 = ResNet50(classes=2, include_top=True, weights=None, input_shape=(224,224,3))
    # compile the resnet model
    resnet_50.compile(optimizer=tensorflow_privacy.DPKerasAdamOptimizer(
        l2_norm_clip=l2,
        noise_multiplier=1.3,
        num_microbatches=1,
        learning_rate = 0.01
    ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    history = resnet_50.fit(sub_x_train,
                        sub_y_train,
                        validation_data=(sub_x_test,sub_y_test),
                        epochs=30,
                        batch_size=32,
                        shuffle=True)
    l2_histories.append(history)

# find the best performing learning rate
lr_histories = []
learning_rate_multiplier = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
for lr in learning_rate_multiplier:
    print("learning_rate: ", lr)
    resnet_50 = ResNet50(classes=2, include_top=True, weights=None, input_shape=(224,224,3))
    # compile the resnet model
    resnet_50.compile(optimizer=tensorflow_privacy.DPKerasAdamOptimizer(
        l2_norm_clip=0.7,
        noise_multiplier=0.3,
        num_microbatches=1,
        learning_rate = lr
    ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    history = resnet_50.fit(sub_x_train,
                        sub_y_train,
                        validation_data=(sub_x_test,sub_y_test),
                        epochs=30,
                        batch_size=32,
                        shuffle=True)
    lr_histories.append(history)
