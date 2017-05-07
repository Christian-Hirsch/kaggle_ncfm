import numpy as np
import os
import pandas as pd
import dill as pickle

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


#seed
SEED = 42
np.random.seed(42)

##############################################
#######FILE PATHS
##############################################
ROOT = '.'

VGG_FEATS = '{}/features/vgg16_block5.npy'.format(ROOT)
DATA_PATH = '{}/data'.format(ROOT)
MODEL_PATH = '{}/models'.format(ROOT)


##############################################
#######PCA and KMeans params
##############################################

#number of pca components
PCA_COMP = int(1e2)

#number of clusters 
N_CLUSTERS = 17 

#number of columns for subsampling
COL_NUMS = int(3e5)

##############################################
##############################################
##############################################
##############################################


# ## Load VGG features and labels
vgg_features = np.load(VGG_FEATS)
vgg_features  = np.reshape(vgg_features, (vgg_features.shape[0],-1))



# Define custom transformer for random column selection

selector = FunctionTransformer(lambda X: np.take(X, np.random.RandomState(42).choice(X.shape[1], int(3e5), replace = False), axis = 1))

# ## PCA
pca = PCA(n_components = PCA_COMP, random_state = SEED)

# ## K-Means clustering 
km = KMeans(n_clusters = N_CLUSTERS, n_jobs = -1, random_state = SEED)

#assemble into pipeline
pipeline = make_pipeline(selector, pca, km)
print(pd.Series(pipeline.fit_predict(vgg_features)).value_counts())



# Persist the fitted kmeans object.
pickle.dump(pipeline, open('{}/boat_clusterer.pkl'.format(MODEL_PATH), 'wb'))

