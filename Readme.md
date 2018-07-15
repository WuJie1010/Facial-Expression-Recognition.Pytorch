# Facial-Expression-Recognition.Pytorch
A CNN based pytorch implementation on facial expression recognition (FER2013 and CK+), achieving 73.112% (state-of-the-art) in FER2013 and 94.64% in CK+

## Dependencies ##
- Python 2.7
- Pytorch >=0.2.0
- h5py (Preprocessing)
- sklearn (plot confusion matrix)

## FER2013 Dataset ##
- Dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
Image Properties: 48 x 48 pixels (2304 bytes)
labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The private test set consists of another 3,589 examples.

### Preprocessing Fer2013 ###
first download the dataset(fer2013.csv) then put it in the "data" folder, then
python preprocess_fer2013.py

### Train and Eval model ###
python mainpro_FER.py --model VGG19 --bs 128 --lr 0.01

### plot confusion matrix ###
python plot_fer2013_confusion_matrix.py --model VGG19 --split PrivateTest

###              fer2013 Accurary             ###
|    Model    | PublicTest_acc | PrivateTest_acc |
|    VGG19    |    71.496%     |     73.112%     |
|   Resnet18  |    71.190%     |     72.973%     |

## CK+ Dataset ##
The CK+ dataset is an extension of the CK dataset. It contains 327 labeled facial videos,
We extracted the last three frames from each sequence in the CK+ dataset, which
contains a total of 981 facial expressions. we use 10-fold Cross validation in the experiment.

### Train and Eval model for a fold ###
python mainpro_CK+.py --model VGG19 --bs 128 --lr 0.01 --fold 1

## Train and Eval model for all 10 fold ###
python k_fold_train.py

### plot confusion matrix for all fold ###
python plot_CK+_confusion_matrix.py --model VGG19

###      CK+ Accurary      ###
|    Model    |   test_acc   |
|    VGG19    |    94.646%   |
|   Resnet18  |    94.040%   |

