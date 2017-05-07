import numpy as np
import os
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#seed
SEED = 42
np.random.seed(42)

##############################################
#######FILE PATHS
##############################################
ROOT = '..'

VGG_FEATS = '{}/features/vgg16_block5.npy'.format(ROOT)
DATA_PATH = '{}/data'.format(ROOT)
FEATURE_PATH = '{}/features'.format(ROOT)

LABEL_PATHS = ['{}/{}.csv'.format(DATA_PATH, label) for label in ['labels', 'train_split', 'val_split']]


##############################################
#######PCA and KMeans params
##############################################

#number of pca components
PCA_COMP = int(1e2)

#number of clusters 
N_CLUSTERS = 250

#number of clusters constituting the validation set
#N_VAL_CLUSTERS = 75

##############################################
##############################################
##############################################
##############################################


# ## Load VGG features and labels
labels = pd.read_csv(LABEL_PATHS[0], index_col = 0)
test_mask = np.isnan(labels['label']).values
labels_test = labels[test_mask]

vgg_features = np.load(VGG_FEATS)[test_mask]
vgg_features = np.reshape(vgg_features, (vgg_features.shape[0],-1))


# Compute PCA on the features.
pca = PCA(n_components = PCA_COMP, random_state = SEED)
pcad_features = pca.fit_transform(vgg_features)


#We cluster the images according to the K-Means algorithm.

km = KMeans(n_clusters = N_CLUSTERS, n_jobs = -1, random_state = SEED)
clust_preds = km.fit_predict(pcad_features)

print(pd.Series(clust_preds).value_counts())

# serialize cluster association.
labels_test['clust_val'] = clust_preds
#labels_test = labels_test['clust_val']
labels_test.to_csv('{}/test_clust.csv'.format(FEATURE_PATH), index = True, header = True)



# serialize train-val-split to disk
#for typ, mask in zip(['train', 'val'], [train_mask, val_mask]):
#    np.savetxt('{}/{}_split.csv'.format(DATA_PATH, typ), labels.index.values[mask], fmt = '%s')


