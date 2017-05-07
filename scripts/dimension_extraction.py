import numpy as np
import os
import pandas as pd

from PIL import Image


##############################################
#######PATH TO DATA
##############################################


ROOT = '.'
DATA_PATH = '{}/data/alldata/tt'.format(ROOT)
FEATURE_PATH = '{}/features'.format(ROOT)


##############################################
##############################################
##############################################
##############################################

#extract sizes
im_dims = np.asarray([[file] + list(Image.open('{}/{}'.format(DATA_PATH, file)). size) for file in os.listdir(DATA_PATH)])

#turn into ordered df
im_df = pd.DataFrame(im_dims, 
 columns = ['image', 'width', 'height']).set_index('image')
im_df.sort_index(inplace = True)

im_df.to_csv('{}/im_dims.csv'.format(FEATURE_PATH), index = True, header = True)
