
import keras
import numpy as np
import os
import pandas as pd

from PIL import Image
from vgg16bn import Vgg16BN

from keras.optimizers import Adam


##############################################
#######PATH TO DATA
##############################################

ROOT = '.'

DATA_PATH = '{}/data/alldata/tt'.format(ROOT)
FEATURE_PATH = '{}/features'.format(ROOT)

##############################################
#######TARGETED IMAGE WIDTH AND HEIGHT
##############################################
IMG_WIDTH = 640
IMG_HEIGHT = 360

##############################################
##############################################
##############################################
##############################################



# ## Load the crop regions
crop_regions = pd.read_csv('{}/crop_features.csv'.format(FEATURE_PATH), index_col = 0)



# We rotate and crop the data suitably/

def crop_rotate_scale_gen():
    count = 0
    while True:
            image = crop_regions.index[count]
            count = count + 1
            img = Image.open('{}/{}'.format(DATA_PATH, image))
            img_row = crop_regions.loc[image]
            
            img = crop(img, img_row)
            img = rotate(img, img_row)
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            yield np.expand_dims(np.moveaxis(np.asarray(img), 2, 0), 0)

def crop(img, img_row):
    return img.crop((img_row['xl'], img_row['yu'], img_row['xr'],img_row['yd']))

def rotate(img, img_row):
    result = img if(img_row['xr'] - img_row['xl'] > img_row['yd'] - img_row['yu']) else img.transpose(Image.ROTATE_90) 
    return result


# ## Neural net

# In[4]:

vgg640 = Vgg16BN((IMG_HEIGHT, IMG_WIDTH)).model
vgg640.pop()
vgg640.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])


#compute features
vgg_features = vgg640.predict_generator(crop_rotate_scale_gen(), val_samples = len(os.listdir(DATA_PATH))
vgg_features = np.moveaxis(vgg_features, 1, 3)

#serialize vgg features
np.save('{}/vgg16_cropped_block5.npy'.format(FEATURE_PATH), vgg_features)

