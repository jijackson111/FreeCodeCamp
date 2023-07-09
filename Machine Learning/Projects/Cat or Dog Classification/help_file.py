from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import skimage.io
from pathlib import Path
import glob
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import itertools
from PIL import Image

# Sort out the dataset and variables
dic = 'cats_and_dogs.zip'
with ZipFile(dic, 'r') as z:
    z.extractall()
PATH = 'cats_and_dogs'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')
labels = ['Cat', 'Dog']

total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 15
target_size = (IMG_WIDTH, IMG_HEIGHT)

# Format function
def input_format(x):
    folder = os.path.join('cats_and_dogs\{}\images'.format(x))
    imgs = Path(folder).glob('*.jpg')
    img_strs = [str(p) for p in imgs]
    return img_strs

train_fn = input_format('train')
test_fn = input_format('test')
valid_fn = input_format('validation')
     

# Lists for the placement of the floats
train_lt = []
for i in range(0, len(train_fn)):
    elem = train_fn[i]
    img = Image.open(elem)
    data = np.asarray(img, dtype=np.float64) 
    train_lt.append(data)
    
# test_lt = []
# for i in range(0, total_test):
#     elem = test_dir[i]
#     image = Image.open(elem)
#     data = np.asarray(image)
#     test_lt.append(data)
    
# valid_lt = []
# for i in range(0, total_val):
#     elem = validation_dir[i]
#     image = Image.open(elem)
#     data = np.asarray(image)
#     valid_lt.append(data)


# Create image generators
train_image_generator = ImageDataGenerator(width_shift_range=IMG_WIDTH,
                              height_shift_range=IMG_HEIGHT,
                              rescale = 1./255,
                              fill_mode='nearest',
                              shear_range=0.2)

train_image_generator.fit(train_lt)