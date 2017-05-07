
import numpy as np
import pandas as pd
import dill as pickle

#seed
SEED = 42
np.random.seed(42)

##############################################
#######FILE PATHS
##############################################
ROOT = '.'

VGG_FEATS = '{}/features/vgg16_block5.npy'.format(ROOT)
LABEL_PATH = '{}/data/labels.csv'.format(ROOT)
FEATURE_PATH = '{}/features'.format(ROOT)
DATA_PATH = '{}/data'.format(ROOT)

##############################################
##############################################
##############################################
##############################################



boat_clusts = pd.read_csv('{}/boat_clust.csv'.format(FEATURE_PATH), index_col = 0)
crop_regs = pd.read_csv('{}/crop_regions_small.csv'.format(DATA_PATH), index_col = 0)

merged_df = boat_clusts.merge(crop_regs, 'left', left_on = 'clust_val', right_index = True)
merged_df = merged_df[['xl', 'yu', 'xr', 'yd']]

merged_df.to_csv('{}/crop_features.csv'.format(FEATURE_PATH), index = True, header = True)
