from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import os.path
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models
from zipfile import ZipFile
from pathlib import Path
import cv2
from time import sleep
from tqdm import tqdm

# Generate progress meter
for i in tqdm(range(10)):
    sleep(3)

# Import and split directories of the dataset,
dic = 'cats_and_dogs.zip'
with ZipFile(dic, 'r') as z:
    z.extractall()
    
PATH = 'cats_and_dogs'
train_dir = os.path.join(PATH, 'train')
val_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')
labels = ['Cat', 'Dog']

# Variables for pre-processing and training.
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
ts = (IMG_WIDTH, IMG_HEIGHT)

# Create data generators
datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory(train_dir)
val_data = datagen.flow_from_directory(val_dir)
test_data = datagen.flow_from_directory(PATH, classes=['test'])

# Function to print out an array of five random images
def plotImages(images_arr, probabilities = False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
    if probabilities is False:
      for img, ax in zip( images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
    else:
      for img, probability, ax in zip( images_arr, probabilities, axes):
          ax.imshow(img)
          ax.axis('off')
          if probability > 0.5:
              ax.set_title("%.2f" % (probability*100) + "% dog")
          else:
              ax.set_title("%.2f" % ((1-probability)*100) + "% cat")
    plt.show()

sample_training_images, _ = next(train_data)
plotImages(sample_training_images[:5])

# Generate augmented images
augmented_images = [train_data[0][0][0] for i in range(5)]
plotImages(augmented_images)

# Compile the model

# Train the model

# Evaluate the Model

# Predictions




