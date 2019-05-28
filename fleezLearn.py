# SO this is our file for Deep learning ze fleez andze beez.
# what we is basically do here is a deep learning prediction model, but we're letting the SVM be
# this is so that we can compare and see which model worketh le best.
# now given the data we CURRENTLY haveth, SVM is probably the better of the two options.
# But now that we know that Dr. Doreenweerd of UH Manoa is working with us on giving us MANY of ze fly photos
# espcially the crispy wing ones IN THE FORMAT WE ASKED FOR, things might change
# YEET

#side not please make sure you have keras installed (do a conda install keras -- eezy)

import pickle
from pathlib import Path
from skimage import io
import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import keras
from keras.models import Sequential

# import Dense, Dropout, Flatten, Conv2D, MaxPooling2D from the keras layers module
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Load the csv file containing labels using pandas
labels = pd.read_csv('photo/fleezData.csv', header = 0)  # <--------------- look here should you change the csv from the default

# print value counts for genus
print(labels.Species.value_counts()) # <--------------- look here should you change the csv from the default

# A List of images here, we're making it dynamically by grabbing it from a folder called photo
image_paths = [Path(item) for i in [glob.glob('photo\\*.%s' % ext) for ext in ["jpg", "jpeg", "tiff", "tif"]] for item in i]

# assign the species label values to y
y = labels.Species.values # <--------------- look here should you change the csv from the default
# initialize standard scaler
ss = StandardScaler()

image_list = []
for i in labels.index:
    # load image
    img = io.imread(image_paths[i]).astype(np.float64)

    # for each channel, apply standard scaler's fit_transform method
    for channel in range(img.shape[2]):
        img[:, :, channel] = ss.fit_transform(img[:, :, channel])

    # append to list of all images
    image_list.append(img)

# convert image list to single array
X = np.array(image_list)

# print shape of X
print(X.shape)