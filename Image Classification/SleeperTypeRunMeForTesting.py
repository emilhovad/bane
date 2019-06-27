# Convolutional Neural Network

# import keras
import numpy as np
import os
# Importing the Keras libraries and packages
# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense
# from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score
# from plot_confusion_matrix import plot_confusion_matrix_new
# import matplotlib.pyplot as plt
import glob
from SleeperTypeCNNTestMethods import RunTestAndGenerateConWithModelFile
from SleeperTypeCNNTestMethods import RunTestAndGenerateConWithModelAndWeightFile
from SleeperTypeInceptionTestMethod import main
import csv



print("Train.")
result = []
#Number of classes
nrOfClasses = 2
amountofTestImages = 0
#location of data set to use
datapathTestImages = '../dataset/FirstSplitToTrainValiAndTest/TrainSetSleeperAugmented'
#classes that are in the data set
#C = {'crossing': 0, 'strait': 1, 'switch_closure_panel': 2}
C = {'concrete': 0, 'wood': 1}
Classes = list(C.keys())
#cound the amount of images in the data set folder
for i in range(nrOfClasses):
    amountofTestImages += len(glob.glob(datapathTestImages + '/' + Classes[i] + '/*.jpg'))

print('Amount Of images in test folder: ' + str(amountofTestImages))
########################### CNN_Baseline ###########################
print('########################### CNN_Baseline ###########################')
#model to use
modelFile = 'ModelsFinal/SleeperTypeCNNBaseline.h5'
#correnspoding weight to the model
weightFile = 'ModelsFinal/SleeperTypeCNNBaseline_weights_epoch_12.h5' 
#part of confusion matrix filename
run =  'TypeOfSleepers'
#title on confusion matrix
what =  'CNN_Baseline-Train'
#Runs the images though the model and create a confusion matrix and returns some performance information about the run 
#(different f1 scores, the confusion matrix with out normalycation and with).
CNNBaselineTrain = RunTestAndGenerateConWithModelFile(modelFile, weightFile, amountofTestImages, datapathTestImages, C, Classes, run, what)
result.append(CNNBaselineTrain)

########################### CNN_High ###########################
print('########################### CNN_High ###########################')
modelFile = 'ModelsFinal/SleeperTypeCNNImproved.h5'
weightFile = 'ModelsFinal/SleeperTypeCNNImproved_weights_epoch_08.h5' 
run =  'TypeOfSleepers_epoch_2'
what =  'CNN_Improved-Train'
CNNHighTrain = RunTestAndGenerateConWithModelFile(modelFile, weightFile, amountofTestImages, datapathTestImages, C, Classes, run, what)
result.append(CNNHighTrain)

########################## Inceptionv3 ###########################
print('########################### Inceptionv3 ###########################')
#model information
RETRAINED_LABELS_TXT_FILE_LOC = "SleeperTyperetrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC = "SleeperTyperetrained_graph.pb"
#location of data set to use (All data in one folder no sub folder)
TEST_IMAGES_DIR = "../dataset/FirstSplitToTrainValiAndTest/TrainSetSleeperAugmentedInception"
#location of what amount to the ground truth if the code should be able to create a confusion matrix 
#(each line in the txt should look like so "image name, class" with out ")
groundTruthtxt = '../dataset/FirstSplitToTrainValiAndTest/TrainSetSleeperAugmentedInception/TrainSetGroundTruth.txt'
#part of confusion matrix filename
run =  'TypeOfSleepers'
#title on confusion matrix
what =  'Inceptionv3-Train'
InceptionHighResult = main(RETRAINED_LABELS_TXT_FILE_LOC, RETRAINED_GRAPH_PB_FILE_LOC, TEST_IMAGES_DIR, groundTruthtxt, C, Classes, run, what)



