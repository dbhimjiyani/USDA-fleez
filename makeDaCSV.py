from pathlib import Path # Used to change filepaths
import os
import glob
import csv
import re
# Check the directory. Just so we know exactly where we is u kno
print(os.getcwd())

# A List of images here, we're making it dynamically by grabbing it from a folder called photo
image_paths = [Path(item) for i in [glob.glob('photo\\*.%s' % ext) for ext in ["jpg", "jpeg", "tiff", "tif"]] for item in i]

def createCSV(path, writer):
    text = re.sub(r'photo\\', '', str(path))
    # Regex for extracting details from name of the file, suhc as the species type, code and the angle of shot
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


with open('photo/fleezData.csv', mode='w') as fleez:
    fleez_writer = csv.writer(fleez, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    fleez_writer.writerow(['FileName', 'FleezId', 'Species', 'View'])
    # for loop over image paths
    for img_path in image_paths:
        file = createCSV(Path(img_path), fleez_writer)
print("youe fool. creating the csv my damn self because nobody helps me in this household")
print()