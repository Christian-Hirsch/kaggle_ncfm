import subprocess

from keras.layers import Activation, BatchNormalization, Convolution2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D
from keras.models import Model, Sequential

##############################################
#######TARGETED IMAGE WIDTH AND HEIGHT
##############################################
MAP_WIDTH = 40
MAP_HEIGHT = 22


##############################################
#########NNET PARAMETERS
##############################################

DROPOUT = 0.4
BATCH_SIZE_FC = 64
FC_SIZE = 64
N_CLASSES = 8



##############################################
##############################################

ROOT = '..'
AWS_ROOT = 's3://kagglencfm/production'

[MODEL_PATH, AWS_MODEL_PATH] = ['{}/models'.format(root) for root in [ROOT, AWS_ROOT]]
MODEL_FNAME = 'vgg16bn_structure.json'

##############################################
##############################################



#define the layers
layers = [Flatten( input_shape = (MAP_HEIGHT, MAP_WIDTH, 512)),
       Dense(FC_SIZE, activation='relu'),
       BatchNormalization(),
       Dropout(DROPOUT),
       Dense(FC_SIZE, activation='relu'),
       BatchNormalization(),
       Dropout(DROPOUT),
       Dense(N_CLASSES, activation='softmax')
]


#turn model to json

lrg_model = Sequential(layers)
model_structure = lrg_model.to_json()


# We persist the fitted model and upload it to s3.
with open("{}/{}".format(MODEL_PATH, MODEL_FNAME), "w") as f:
 f.write(model_structure)

command = "aws s3 cp {}/{} {}/{}".format(MODEL_PATH, MODEL_FNAME, AWS_ROOT, MODEL_FNAME)
subprocess.Popen(command.split(" "))

