NCFM Competition
================

What is the difference between a yellow-fin tuna and an albacore tuna? In this competition, deep learning is leveraged to perform this classification task. Thus, Kagglers contribute to reduce illegal fishing

Objective
--
The fish on the fishing-boat images is to be assigned to 1 of 8 types. The score is computed using categorical_crossentropy.

Data
--
The train set consists of 3777 images of fish. The image sizes may vary from boat to boat, but are roughly of the order 1280 x 720. 

Word of caution
--
I didn't have sufficient amount of time for turning my approach into the fully reproducible pipeline needed to satisfy the model submission requirement. Hence, my final submission is based on a less sophisticated involving ensembles 

Preprocessing 
================================================================

File structure initializiation
-
The file structure is initialized via `file_structure_init.py'.

Label extraction
-
The labels are extracted via`label_extraction.py` and are serialized in `data/labels.csv` . Test images have missing labels.


Validation set generation
================================================================

VGG16 feature extraction
-
The features from the last conv-block of the batch-normalized VGG16 from Jeremy Howard are extracted via `vgg_features.py` and serialized in `features/vgg16_block5_features.csv`.


Validation set
-
In the NCFM competition it is challenging to set up a good validation strategy, since the training set contains many highly similar images. We use a clustering based on the level-5 VGG features in order separate similar images from non-similar ones. The split is computed in `train_val_split.py` and serialized in `data/train_split.csv` as well as `data/val_split.csv`

Feature extraction
===============================================================

Image size extraction
-
The image sizes are extracted via `dimension_extraction.py` and are serialized in `features/im_dims.csv`.


Boat classification
-
First, we determine a clustering into 17 boats. A K-Means clustering is fitted in `boat_clust_fit.py` and the clusterer is persisted as `features/boat_clusterer.pkl`. Clusters can be predicted using `boat_clust_predict.py` and are serialized as `features/boat_clust.csv`. In `data/crop_regions_small.csv` we have recorded suggested cropping coordinates for each of the clusters. Then `crop_reg_predict.py` serializes the crop-regions for each file into `data/crop_regions.csv`.


VGG16 cropped feature extraction
-
We extract the VGG16 features from the cropped pictures via `vgg_cropped_features.py`. The features are serialized in `features/vgg16_cropped_block5_features.csv`.




Neural networks 
===============================================================

Define architecture for FCN
-
The FCN network suggested by Jeremy Howard is constructed via `fcn_structure.py`. The model is serialized in `models/fcn_structure.json`. Similarly, we construct a simple nn via `simpleNN_structure.py`. Finally, we use the batch-normalized VGG16 suggested by Jeremy Howard, see`vgg16bn_structure.py` .


Fit weights
-
The weights for a given network structure are fitted with `train_network.py`. The models for network `x_structure.json` are persisted in `x_weights.h5`.


YOLO
===============================================================

save cropped images
-
The cropped images are saved with `cropped_images.py`. They are located in `data/cropped_images/JPEGImages`.


generate yolo bb
-
generate the yolo bb with `yolo_localization.py`. They are saved in `data/cropped_images/labels`.


generate train file
-
generate the yolo train file with `train-test.py`


extract bounding boxes
-
extract bounding boxes using YOLO with `bb_predictions.py`. Predictions are saved in `bb_preds` and `bb_preds_cropped`.


crop yolo bounding boxes
-
We crop the bounding boxes from yolo using `yolo-crop.py` and save them to `data/processed/cropped_data`.



Test Augmentation
===============================================================


Test image grouping
-
TBA
We cluster similar test images based on their VGG features. This is implemented in `test_clust_split.py` and the cluster organization is persisted in 
We use a clustering based on the level-5 VGG features in order separate similar images from non-similar ones. The split is computed in `train_val_split.py` and serialized in `data/train_split.csv` as well as `data/val_split.csv`

First, we determine a clustering into 17 boats. A K-Means clustering is fitted in `boat_clust_fit.py` and the clusterer is persisted as `features/boat_clusterer.pkl`. Clusters can be predicted using `boat_clust_predict.py` and are serialized as `features/boat_clust.csv`. In `data/crop_regions_small.csv` we have recorded suggested cropping coordinates for each of the clusters. Then `crop_reg_predict.py` serializes the crop-regions for each file into `data/crop_regions.csv`.





Submission 
================================================================
Submission 
-
Generate submission file from a trained model with `generate_submission.py`.


Mispredictions 
================================================================
Mispredictions 
-
The mispredictions of a trained classifier are extracted via `mispredictions.py`.



Bounding box extraction
-
For each image we extract the coordinates of the largest bounding box. 


