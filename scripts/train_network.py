import h5py
import numpy as np
import pandas as pd
import os
import subprocess

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Activation, BatchNormalization, Convolution2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D
from keras.models import Model, Sequential, model_from_json, load_model

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

SEED = 42
np.random.seed(SEED)

##############################################
##############################################

ROOT = '..'
AWS_ROOT = 'http://kagglencfm/production'

VGG_FEATS = '{}/features/vgg16_cropped_block5.npy'.format(ROOT)
DATA_PATH = '{}/data'.format(ROOT)
[MODEL_PATH, AWS_MODEL_PATH] = ['{}/models'.format(root) for root in [ROOT, AWS_ROOT]]
 

MODEL_FNAME = 'vgg16bn'

##############################################
##############################################

#parameters for fc layer
BATCH_SIZE = 64

#number of training epochs
NB_EPOCH = 6

#initial learning rate
LR_0 = 1e-3

#Parameters for Learning rate reduction scheme
MIN_LR = 1e-5
LR_PATIENCE = 2
FACTOR = .1

#Parameters for early stopping and checkpoint
EARLY_STOP_PATIENCE = 10
CP_PATH = "{}/{}".format(MODEL_PATH, MODEL_FNAME) + "-{epoch:02d}-{val_loss:.2f}_weights.h5"

#Should we train on the full data?
TRAIN_FULL = True


###############################################
###############################################
###############################################
###############################################


#load network
lrg_model = model_from_json(open("{}/{}_structure.json".format(MODEL_PATH, MODEL_FNAME), 'r').read())
lrg_model.compile(Adam(lr = LR_0), loss = 'categorical_crossentropy', metrics=['accuracy'])



# Load labels and encode labels.
labels_df = pd.read_csv('{}/labels.csv'.format(DATA_PATH), index_col = 0)
idxs = labels_df.index.values
labels = labels_df['label'].values


# test-train split.
tv_idxs = [pd.read_csv('{}/{}_split.csv'.format(DATA_PATH, typ), header = None)[0].values
                          for typ in ['train', 'val']]
tv_masks = [np.in1d(idxs, typ) for typ in tv_idxs]
tv_masks_triple= [tv_masks[0] + tv_masks[1], tv_masks[0], tv_masks[1]]





# Next, load vgg_features and labels
vgg_feats = np.load(VGG_FEATS)

#We subdivide into training and validation data
lab_encs_triple = [OneHotEncoder(sparse = False).fit_transform(labels[mask].reshape(-1, 1))  for mask in tv_masks_triple]
vgg_feats_triple = [vgg_feats[mask] for mask in tv_masks_triple]


if(TRAIN_FULL):
 
 lrg_model.fit(vgg_feats_triple[0], lab_encs_triple[0], batch_size = BATCH_SIZE, nb_epoch = NB_EPOCH)


else:

 #Iinitial fitting neural network
 lrg_model.fit(vgg_feats_triple[1], lab_encs_triple[1], batch_size = BATCH_SIZE, nb_epoch = 1,
             validation_data = (vgg_feats_triple[2], lab_encs_triple[2]))


 #Add callbacks for LR-descrease scheme, early stopping and checkpointing

 reduce_lr = ReduceLROnPlateau(monitor =  'val_loss', factor = FACTOR, patience = LR_PATIENCE, min_lr = MIN_LR, verbose=1)
 estop = EarlyStopping(monitor='val_loss', patience = EARLY_STOP_PATIENCE, verbose = 1)
 checkpoint = ModelCheckpoint(CP_PATH, monitor='val_loss', verbose = 1, save_best_only=True)


 #Main fitting of network
 lrg_model.optimizer.lr.assign(LR_0)
 lrg_model.fit(vgg_feats_triple[1], lab_encs_triple[1], batch_size = BATCH_SIZE, callbacks = [reduce_lr, estop, checkpoint], 
              nb_epoch = NB_EPOCH, validation_data = (vgg_feats_triple[2], lab_encs_triple[2]))

#save fitted weights
#lrg_model.save('{}/{}_weights.h5'.format(MODEL_PATH, MODEL_FNAME))

