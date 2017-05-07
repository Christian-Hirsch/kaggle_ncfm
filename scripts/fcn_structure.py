import subprocess

from keras.layers import Activation, BatchNormalization, Convolution2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D
from keras.models import Model, Sequential

##############################################
##############################################

ROOT = '..'
AWS_ROOT = 's3://kagglencfm/production'

[MODEL_PATH, AWS_MODEL_PATH] = ['{}/models'.format(root) for root in [ROOT, AWS_ROOT]]
MODEL_FNAME = 'fcn_structure.json'

##############################################
##############################################

#parameters for fc layer
BATCH_SIZE_FC = 64
DROPOUT = 0.0
SIZE_FC = 128


#define the layers
layers = [
        BatchNormalization(axis=-1, input_shape = (22, 40, 512)),
        Convolution2D(SIZE_FC,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=3),
        MaxPooling2D(),
        Convolution2D(SIZE_FC,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=-1),
        MaxPooling2D(),
        Convolution2D(SIZE_FC,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis = -1),
        MaxPooling2D((1,2)),
        Convolution2D(8,3,3, border_mode='same'),
        Dropout(DROPOUT),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ]


#turn model to json

lrg_model = Sequential(layers)
model_structure = lrg_model.to_json()


# We persist the fitted model and upload it to s3.
with open("{}/{}".format(MODEL_PATH, MODEL_FNAME), "w") as f:
    f.write(model_structure)

command = "aws s3 cp {}/{} {}/{}".format(MODEL_PATH, MODEL_FNAME, AWS_ROOT, MODEL_FNAME)
subprocess.Popen(command.split(" "))

