# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json
import random
from typing import Tuple
from scipy import special
import keras.backend as K

from torchvision import transforms
import torch
import torch.nn.functional as F
import torchvision
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from sklearn import metrics
from sklearn.model_selection import train_test_split

import PIL
from PIL import Image

from keras.applications.resnet50 import ResNet50

import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData, AttackResultsCollection, AttackType, PrivacyMetric, PrivacyReportMetadata, SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import privacy_report

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

class PrivacyMetrics(tf.keras.callbacks.Callback):
  def __init__(self, epochs_per_report, model_name):
    self.epochs_per_report = epochs_per_report
    self.model_name = model_name
    self.attack_results = []

  def on_epoch_end(self, epoch, logs=None):
    epoch = epoch+1

    if epoch % self.epochs_per_report != 0:
      return

    print(f'\nRunning privacy report for epoch: {epoch}\n')

    logits_train = self.model.predict(x_train, batch_size=batch_size)
    logits_test = self.model.predict(x_test, batch_size=batch_size)

    prob_train = special.softmax(logits_train, axis=1)
    prob_test = special.softmax(logits_test, axis=1)

    # Add metadata to generate a privacy report.
    privacy_report_metadata = PrivacyReportMetadata(
        # Show the validation accuracy on the plot
        # It's what you send to train_accuracy that gets plotted.
        accuracy_train=logs['val_accuracy'],
        accuracy_test=logs['val_accuracy'],
        epoch_num=epoch,
        model_variant_label=self.model_name)

    attack_results = mia.run_attacks(
        AttackInputData(
            labels_train=y_train,
            labels_test=y_test,
            probs_train=prob_train,
            probs_test=prob_test),
        SlicingSpec(entire_dataset=True, by_class=True),
        attack_types=(AttackType.THRESHOLD_ATTACK,
                      AttackType.LOGISTIC_REGRESSION),
        privacy_report_metadata=privacy_report_metadata)

    self.attack_results.append(attack_results)

class CategoricalTruePositives(tf.keras.metrics.Metric):

    def __init__(self, num_classes, batch_size,
                 name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)

        self.batch_size = batch_size
        self.num_classes = num_classes

        self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_test, y_pred, sample_weight=None):

        y_test = K.argmax(y_test, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)
        y_test = K.flatten(y_test)

        true_poss = K.sum(K.cast((K.equal(y_test, y_pred)), dtype=tf.float32))

        self.cat_true_positives.assign_add(true_poss)

    def result(self):

        return self.cat_true_positives

