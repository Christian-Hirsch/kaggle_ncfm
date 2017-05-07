
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
MODEL_PATH = '{}/models'.format(ROOT)

##############################################
##############################################
##############################################
##############################################




# First, we load the VGG16 features.
vgg_features = np.load(VGG_FEATS)
vgg_features = np.reshape(vgg_features, (vgg_features.shape[0],-1))


# restore pickled classifier
km = pickle.load(open('{}/boat_clusterer.pkl'.format(MODEL_PATH), 'rb'))
clust_preds = km.predict(vgg_features)


# serialize cluster association.
labels = pd.read_csv(LABEL_PATH, index_col = 0)
labels['clust_val'] = clust_preds
labels = labels['clust_val']
labels.to_csv('{}/boat_clust.csv'.format(FEATURE_PATH), index = True, header = True)

