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
from skimage.transform import resize
import pandas as pd
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import keras
from keras.models import Sequential

# import Dense, Dropout, Flatten, Conv2D, MaxPooling2D from the keras layers module
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Load the csv file
labels = pd.read_csv('photo/fleezData.csv', header = 0)  # <--------------- look here should you change the csv from the default

# print value counts for species number
print(labels.FileName.value_counts()) # <--------------- look here should you change the csv from the default

# A List of images here, we're making it dynamically by grabbing it from a folder called photo
image_paths = [Path(item) for i in [glob.glob('photo\\*.%s' % ext) for ext in ["jpg", "jpeg", "tiff", "tif"]] for item in i]

# assign the species label values to y
y = labels.Fleez.values # <--------------- look here should you change the csv from the default
# initialize standard scaler
ss = StandardScaler()

def get_image(path):
    img = Image.open(path)
    image = img.resize((50, 50), Image.LANCZOS)  # <--------------- look here should you change the csv from the default
    return np.array(image)

image_list = []
for i in labels.index:
    # load image
    ''' This needs to call separate function isntead of doing an io.imread here for some reason as we aren't able to get
    the correct matrix shape for if don't keep the two processes separate. Really bizarre but I mean hey, it works '''
    img = get_image(image_paths[i]).astype(np.float64)

    # for each channel, apply standard scaler's fit_transform method
    for channel in range(img.shape[2]):
        img[:, :, channel] = ss.fit_transform(img[:, :, channel])

    # append to list of all images
    image_list.append(img)

# convert image list to single array
X = np.array(image_list)

# print shape of X
''' to make sure we are on the right track. This should read as four column values with the last value being
 3 (that is the value for the number of color channels that exist in our image list). IMPORTANT: if only one value
 shows up, that means that there isn't a standard pixel size of the image list, and the matrix has taken everything in
 as one giant column of array entry data. Not good as the model will crash if that is the case. '''
print(X.shape)

# split out evaluation sets (x_eval and y_eval)
x_interim, x_eval, y_interim, y_eval = train_test_split(X,
                                           y,
                                           test_size=0.2, # percent of our data that we want to use for testing the prediction model
                                           random_state=52)

# split remaining data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_interim,
                                           y_interim,
                                           test_size=0.4,
                                           random_state=52)

# examine number of samples in train, test, and validation sets
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_eval.shape[0], 'eval samples')

# set model constants
num_classes = 1

# define model as Sequential
model = Sequential()

# first convolutional layer with 32 filters
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)))
# add a second 2D convolutional layer with 64 filters
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# reduce dimensionality through max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# third convolutional layer with 64 filters
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# add dropout to prevent over fitting
model.add(Dropout(0.25))
# necessary flatten step preceeding dense layer
model.add(Flatten())
# fully connected layer
model.add(Dense(128, activation='relu'))

# add additional dropout to prevent overfitting
model.add(Dropout(0.5))

# prediction layers
model.add(Dense(num_classes, activation="sigmoid", name='preds'))

# show model summary
model.summary()


# Now we compile and train model
model.compile(
    # set the loss as binary_crossentropy
    loss=keras.losses.binary_crossentropy,
    # set the optimizer as stochastic gradient descent
    optimizer=keras.optimizers.SGD(lr=0.001),
    # set the metric as accuracy
    metrics=['accuracy']
)

# mock-train the model using the first ten observations of the train and test sets
model.fit(
    x_train[:10, :, :, :],
    y_train[:10],
    epochs=5,
    verbose=1,
    validation_data=(x_test[:10, :, :, :], y_test[:10]) #hnnGH validate me zaddy
)