import keras
import numpy as np
import os
import pandas as pd
from vgg16bn import Vgg16BN

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator




##############################################
#######PATH TO DATA
##############################################


ROOT = '.'

DATA_PATH = '{}/data/alldata'.format(ROOT)
VGG_FEATS = '{}/features'.format(ROOT)


##############################################
#######TARGETED IMAGE WIDTH AND HEIGHT
##############################################
IMG_WIDTH = 640
IMG_HEIGHT = 360


##Image generator
gens = ImageDataGenerator().flow_from_directory(
        DATA_PATH,
        target_size = (IMG_HEIGHT, IMG_WIDTH),
        batch_size = 1,
        shuffle = False,
        class_mode = None) 

##Neural Net
vgg640 = Vgg16BN((IMG_HEIGHT, IMG_WIDTH)).model
vgg640.pop()
vgg640.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])

filenames = [file.split('/')[1] for file in  gens.filenames]
argsort = np.argsort(filenames)


#sort vgg_features
vgg_features = np.moveaxis(vgg640.predict_generator(gens, len(filenames)), 1, 3)
vgg_features = np.take(vgg_features, argsort, axis = 0)

np.save('{}/vgg16_block5.npy'.format(VGG_FEATS), vgg_features) 
