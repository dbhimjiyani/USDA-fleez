Aloha,

My name is Devanshi and I am working with the USDA in Hawai\`i (in collaboration with the scientists at the University of Hawai`i at Manoa) to build 
an image recognition and prediction model for recognizing fruitfly species.
This project is intended to be an open source research software, and we hope that with it, recognizing pest fruit fly species that are wreaking 
absolute havoc on island crops, would be easier. The need for this software being that the pest species are phenotypically verrryyy similar to 
sister species that are local. We are hoping to build this software so that even if you aren't a biologist with a glittering PhD, should you need to check the species of a fruitfly, all you would have to do is tear off the wings (sadface) and stick it under clearn tape and a microscope 
(if can) and just scan a picture on your phone and the app could tell if you should warn your local Ag dept. a new potential threat to the flora 
of your area's ecosystem. Pretty neat stuff.
  
For the development of this software, I am using Python (via PyCharm) as a primary language, and have built two different prediction models.
One is a simple Support Vector Machine model, and has an prediction accuracy score that is erm, on the lower end of things. This is saved in the file named fleezSVM.
I have made another model which is a Deep learning CNN based model. This one is in a file called fleezLearn, and has a far better accuracy. Right now it is just in one file,
but I plan on splitting this into two files, one for learning and training and one for predicting. This file is called fleezPredict. 
The model I made in fleezLearn has been saved in two files, one with a .h5 ext which I use, and one unupdated YAML in case someone needs things in YAML form.

Additionally, we are also set to recieve a sick new collection of wing images from UH Manoa scientists, as soon as they're done clicking and
naming them. That's right, fresh off the block straight for us. YEET.

Mahalo and please feel free to contact me at devanshi.bhimjiyani@gmail.com if you have any questions/suggestions!


Sincerely,

Devanshi