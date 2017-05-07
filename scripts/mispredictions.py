import numpy as np
import os
import pandas as pd

from keras.models import load_model


##############################################
#######FILE PATHS
##############################################
ROOT = '..'
AWS_ROOT = 'http://kagglencfm/production'

VGG_FEATS = '{}/features/vgg16_cropped_block5.npy'.format(ROOT)
DATA_PATH = '{}/data'.format(ROOT)
OUTPUT_PATH = '{}/../data/interim'.format(ROOT)

MODEL_PATH = '{}/models'.format(ROOT)
MODEL_FNAME = 'fcn-01-0.87_weights.h5'

FISH_NAMES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

##############################################
##############################################
##############################################
##############################################

# First, we load the labels

labels_df = pd.read_csv('{}/labels.csv'.format(DATA_PATH), index_col = 0)
idxs = labels_df.index
labels = labels_df['label'].values

#we extract the validation set as mask
val_idxs = pd.read_csv('{}/val_split.csv'.format(DATA_PATH), header = None)[0].values
val_mask = np.in1d(idxs, val_idxs)
val_idxs = idxs[val_mask]


# Next, we load the unpooled VGG16 features from block 5. 
vgg_features = np.load(VGG_FEATS)[val_mask]


# load the network and generate predictions
model = load_model("{}/{}".format(MODEL_PATH, MODEL_FNAME))
preds = model.predict_classes(vgg_features)
pred_proba = model.predict_proba(vgg_features)


#generate the error mask
error_mask = (preds != labels[val_mask])


#use error mask to filter probabilities, labels and filenames
false_predictions = pred_proba[error_mask]
correct_label = labels[val_mask][error_mask]
fnames = val_idxs[error_mask]

#create data frame and persist to disk
error_df = pd.DataFrame(false_predictions, columns = FISH_NAMES, index = fnames)
error_df['label'] = correct_label
error_df.to_csv('{}/wrong_preds.csv'.format(OUTPUT_PATH), header = True, index = True)


