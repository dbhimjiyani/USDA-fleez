from pathlib import Path # Used to change filepaths
import os
import glob
# We set up matplotlib, pandas, and the display function
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
import numpy as np
import csv
import re

# import Image from PIL so we can use it later
from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2grey

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# import train_test_split from sklearn's model selection module
from sklearn.model_selection import train_test_split

# import SVC from sklearn's svm module
from sklearn.svm import SVC

# import accuracy_score from sklearn's metrics module
from sklearn.metrics import roc_curve, auc, accuracy_score

# Check the directory. Just so we know exactly where we is u kno
print(os.listdir())

# A List of images here, we're making it dynamically by grabbing it from a folder (here it is photo)
image_paths = [Path(item) for i in [glob.glob('photo\\*.%s' % ext) for ext in ["jpg", "jpeg", "tiff", "tif"]] for item in i]

# load the csv file containing labels using pandas
labels = pd.read_csv('saved/fleezData.csv', index_col=1)

# show the first five rows of the dataframe using head
display(labels.head())

def get_image(path):
    img = Image.open(path)
    return np.array(img)

def create_features(img):
    # flatten three channel color image
    color_features = img.flatten()
    # convert image to greyscale
    grey_image = rgb2grey(img)
    # get HOG features from greyscale image
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack((color_features, hog_features))
    return flat_features


def create_feature_matrix():
    features_list = []

    for im in image_paths:
        # load image
        print("getting img")
        img = get_image(im)
        print(str(im))
        # get features for image
        print("gettin da features, brah")
        image_features = create_features(img)
        print(image_features.shape)
        features_list.append(image_features)
        print("appEND")

    # convert list of arrays into a matrix
    print("'features list length '")
    print(len(features_list))
    feature_matrix = np.array(features_list)

    return feature_matrix

# # run create_feature_matrix on our dataframe of images
try:
    if os.path.getsize('saved/fleezData.csv') > 0:
        print("Entering ze luup")
        feature_matrix = create_feature_matrix()

        # get shape of feature matrix
        print('Feature matrix shape is: ', feature_matrix.shape)

    else:
        def createCSV(path, writer):
            text = re.sub(r'photo\\', '', str(path))
            matchObj = re.match(
                r'([a-zA-Z0-9]+|(B\_[a-zA-Z0-9\-]+\_[a-zA-Z0-9\-]+)|([a-zA-Z0-9\-]+\_[a-zA-Z0-9\-]+)(\_CostaNotExp)*)\_((ms|FF)[0-9\-A-Za-z]+(_ZN)*)\_([a-z])',
                text)
            try:
                species = matchObj.group(1)
                fleezId = matchObj.group(5)
                view = matchObj.group(8)
            except AttributeError:
                species = ""
                fleezId = ""
                view = ""
            writer.writerow([text, fleezId, species, view])


        with open('saved/fleezData.csv', mode='w') as fleez:
            fleez_writer = csv.writer(fleez, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            fleez_writer.writerow(['FileName', 'FleezId', 'Species', 'View'])
            # for loop over image paths
            for img_path in image_paths:
                createCSV(Path(img_path), fleez_writer)
        print("youe fool. Entering ze luup after creating the csv my damn self because nobody helps me in this household")
        feature_matrix = create_feature_matrix()
except OSError as e:
    print("Corrupt File/Doest Not Exist")

# define standard scaler
ss = StandardScaler()
# run this on our feature matrix
fleez_stand = ss.fit_transform(feature_matrix)

pca = PCA(n_components=500)
# use fit_transform to run PCA on our standardized matrix
fleez_pca = pca.fit_transform(fleez_stand)
print('PCA matrix shape is: ', fleez_pca.shape)

# displaying the HOG of a single, final, greyscaled image
'''# run HOG using our greyscale bombus image
    gs_path = "saved/gs_{}.jpg".format(path.stem)
    gs = img.convert(mode="L")
    gs.save(gs_path)
    hog_features, hog_image = hog(final,
                                  visualize=True,
                                  block_norm='L2-Hys',
                                  pixels_per_cell=(16, 16))

    # show our hog_image with a grey colormap
    # Allows the pop up window to come up
    plt.show(block=True)
    plt.imshow(hog_image, cmap=mpl.cm.gray)'''

# Single image experiment
'''
# Grab the image
img = Image.open('photo\\ms7797_Bactrocera_nigrita_abdomen.tif')

# Get the image size
img_size = img.size
print("The image size is: {}".format(img_size))
#img.show()

# Turn our image object into a NumPy array

# Convert to grayscale and then to numpy array
img_bw = img.convert(mode="L")
img_data = np.array(img_bw)
img_data1 = np.array(img)

# Get the shape of the resulting array
img_data_shape = img_data.shape

print("Our NumPy array has the shape: {}".format(img_data_shape))

# Plot the data with `imshow`
plt.imshow(img_data, cmap=plt.cm.gray)
plt.show()


# Create higher contrast by reducing range
hc = np.maximum(img_data, 100)

# Show the higher contrast version
plt.imshow(hc, cmap=plt.cm.gray)
plt.show()

# Convert the NumPy array to an Image to save in a new folder called Saved
hcImg = Image.fromarray(hc)
bwImg = Image.fromarray(img_data)

# Save the high contrast version
hcImg.save("saved/bw_hc_bactrocera_nigrita_abdomen.jpg")
bwImg.save("saved/bw_bactrocera_nigrita_abdomen.jpg")
'''

# Making sure your plt viewing is fine
'''
#----------------------------------------------------
#generate test_data
test_data = np.random.beta(a=1, b=1, size=(100, 100, 3))
#display the test_data
plt.title('Test Image - Created in our code!')
# Allows the pop up window to come up
plt.show(block=True)
plt.imshow(test_data)
#-----------------------------------------------------
'''
'\n'