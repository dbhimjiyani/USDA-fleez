from pathlib import Path # Used to change filepaths
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# A List of images here, we're making it dynamically by grabbing it from a folder called photo
image_paths = [Path(item) for i in [glob.glob('photo\\*.%s' % ext) for ext in ["jpg", "jpeg", "tiff", "tif"]] for item in i]

# Load the csv file containing labels using pandas
labels = pd.read_csv('photo/fleezData.csv', header = 0)  # <--------------- look here should you change the csv from the default

# Get image, resize then return in an array form
'''Get the image from a given path, 
resize to a 2:1 ratio for even num of px in all of them,
return it in a numpy array'''
def get_image(path):
    img = Image.open(path)
    image = img.resize((200, 100), Image.LANCZOS)  # <--------------- look here should you change the csv from the default
    return np.array(image)

# create features from a numpy array of an image
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

# Create a matrix of the features
'''Feed a path into get_image, get the numpy array,
 then feed that into create_features to process,
 then append onto on one giant array that is then made into a matrix.
 NOTE: For this to work properly into the next step, you must have the same number of features in EVERY
 SINGLE IMAGE in your folder. Hence the resize in the get_image method'''
def create_feature_matrix():
    features_list = []

    for im in image_paths:
        # load image
        img = get_image(im)
        # get features for image
        image_features = create_features(img)
        features_list.append(image_features)

    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)

    return feature_matrix

# Run create_feature_matrix on our dataframe of images
try:
    if os.path.getsize('photo/fleezData.csv') > 0:  # <--------------- should you change the csv from the default
        print("Entering ze luup")
        feature_matrix = create_feature_matrix()

        # get shape of feature matrix
        print('Feature matrix shape is: ', feature_matrix.shape)
except OSError as e:
    print("Corrupt File/Doest Not Exist")

# define standard scaler
ss = StandardScaler()
# run this on our feature matrix
fleez_stand = ss.fit_transform(feature_matrix)
pca = PCA(n_components=360)  # <--------------- should you change the csv from the default
# use fit_transform to run PCA on our standardized matrix
fleez_pca = pca.fit_transform(fleez_stand)
print('PCA matrix shape is: ', fleez_pca.shape)

# Split data into train and test sets here. Test size is 30% of the total data
X_train, X_test, y_train, y_test = train_test_split(fleez_pca,
                                                    labels['Species'].values,  # <--------------- look here should you change the csv from the default
                                                    test_size=.3,
                                                    random_state=1234123)

# look at the distrubution of labels in the train set
print(pd.Series(y_train).value_counts())

# define support vector classifier
svm = SVC(kernel='linear', probability=True, random_state=42)

# fit model
svm.fit(X_train, y_train)

# generate predictions
y_pred = svm.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_pred, y_test)
print('Model accuracy is: ', accuracy)

# predict probabilities for X_test using predict_proba
probabilities = svm.predict_proba(X_test)

# select the probabilities for label 1.0
y_proba = probabilities[:, 1]

# calculate false positive rate and true positive rate at different thresholds
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)

# calculate AUC
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
# plot the false positive rate on the x axis and the true positive rate on the y axis
roc_plot = plt.plot(false_positive_rate,
                    true_positive_rate,
                    label='AUC = {:0.2f}'.format(roc_auc))

plt.legend(loc=0)
plt.plot([0,1], [0,1], ls='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate');
plt.show()

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