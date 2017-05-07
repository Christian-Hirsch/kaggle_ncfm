import h5py
import numpy as np
import pandas as pd

from keras.models import load_model 

SEED = 42
np.random.seed(SEED)

##############################################
##############################################

ROOT = '..'
AWS_ROOT = 'http://kagglencfm/production'

VGG_FEATS = '{}/features/vgg16_cropped_block5.npy'.format(ROOT)
DATA_PATH = '{}/data'.format(ROOT)
MODEL_PATH = '{}/models'.format(ROOT) 

SUBMISSION_PATH = '{}/../submissions'.format(ROOT)


MODEL_FNAME = 'vgg16bn_weights.h5'
SUBM_FNAME = 'best_vgg16bn.csv.gz'

FISH_COLUMNS = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']

##############################################
########FLOAT FORMAT FOR SUBMISSION
##############################################

CLIP = 5e-3
FLOAT_FORMAT = '%.5f'

##############################################
##############################################

#load network
model = load_model("{}/{}".format(MODEL_PATH, MODEL_FNAME))


#extract test labels
labels_df = pd.read_csv('{}/labels.csv'.format(DATA_PATH), index_col = 0)
idxs = labels_df.index
labels = labels_df['label'].values
test_mask = np.isnan(labels)

# Next, load vgg_features and take only the test set
vgg_feats = np.load(VGG_FEATS)[test_mask]

#generate predictions and write into dataframe
preds = model.predict_proba(vgg_feats)
preds_df = pd.DataFrame(preds, columns = FISH_COLUMNS, index = idxs[test_mask])
preds_df = preds_df.clip(lower = CLIP)

#persist predictions
preds_df.to_csv('{}/{}'.format(SUBMISSION_PATH, SUBM_FNAME), compression = 'gzip', header = True, index = True, float_format = FLOAT_FORMAT)
