import os
import pandas as pd
import numpy as np
import PIL
from PIL import Image
import subprocess

#ROOT = "/home/ubuntu/kagglencfm/data"
ROOT = ".."
FEATURE_PATH = "{}/features".format(ROOT)
DATA_PATH = "{}/data/alldata/tt".format(ROOT)
DESTINATION = '{}/data/cropped_iges/JPEGImages'.format(ROOT)

def crop_rotate_save_image(row):
    img = Image.open('{}/{}'.format(DATA_PATH, row.name))
    img = img.crop((row['xl'], row['yu'], row['xr'], row['yd']))
    img = img if(row['xr'] - row['xl'] > row['yd'] - row['yu']) else img.transpose(Image.ROTATE_90)
    img.save('{}/{}'.format(DESTINATION, row.name))


crop_df = pd.read_csv('{}/crop_features.csv'.format(FEATURE_PATH), index_col = 0)
crop_df.apply(crop_rotate_save_image, axis = 1)


