import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder


##############################################
#######PATH TO DATA
##############################################


ROOT = '.'
DATA_PATH = '{}/data/'.format(ROOT)
TRAIN_TEST_PATH = ['{}/{}/'.format(DATA_PATH, path) for path in ['train','test']]


##############################################
#######MAIN PROGRAM
##############################################

labels_list = [[file, fish_type] for path in TRAIN_TEST_PATH for fish_type in os.listdir(path) for file in os.listdir(os.path.join(path, fish_type))]

labels_series = pd.DataFrame(labels_list, columns = ['image', 'label']).set_index('image')
test_mask = labels_series['label'] == 'ZZ'

#set test labels to nan
labels_series['label'] = LabelEncoder().fit_transform(labels_series['label'])
labels_series['label'] = np.where(test_mask, np.nan, labels_series['label'])
labels_series.sort_index(inplace = True)


#serialize 
labels_series.to_csv('{}/labels.csv'.format(DATA_PATH), header = True, index = True)
