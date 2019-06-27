# Convolutional Neural Network

import keras
import numpy as np
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from plot_confusion_matrix import plot_confusion_matrix_new
from plot_confusion_matrix import calculateRecallPrecisionAndF1Score
import matplotlib.pyplot as plt

def RunTestAndGenerateConWithModelFile(modelFile, weightFile, amountofTestImages, datapathTestImages, C, Classes, run, what):
    PredictedandGroundTruthList = []

    test_datagen = ImageDataGenerator(rescale = 1./255)

    test_set = test_datagen.flow_from_directory(datapathTestImages,
                                                target_size = (256, 256),
                                                batch_size = amountofTestImages,
                                                color_mode = "grayscale")


    model = load_model(modelFile)
    model.load_weights(weightFile)
    #model = load_model('CNN_Model_LowRes_RailType10Epochs_8000StepsPreEpochs.h5')


    for i in range(amountofTestImages):
        image, label = test_set._get_batches_of_transformed_samples(np.array([i]))
        #image = test_set._get_batches_of_transformed_samples(np.array([i]))
        image_name = test_set.filenames[i]
        pred = model.predict_classes(image)
        tmp = image_name.split('\\')[-1]
        if label[0][0] == 1:
            PredictedandGroundTruthList.append(str(tmp[:-4]) + ',' + Classes[pred[0]] + "," + Classes[0] + "\n")
        elif label[0][1] == 1:
            PredictedandGroundTruthList.append(str(tmp[:-4]) + ',' + Classes[pred[0]] + "," + Classes[1] + "\n")
        elif label[0][2] == 1:
            PredictedandGroundTruthList.append(str(tmp[:-4]) + ',' + Classes[pred[0]] + "," + Classes[2] + "\n") 
    
    y_true = []
    y_pred = []
    
    for i in range(len(PredictedandGroundTruthList)):
        tmp = PredictedandGroundTruthList[i].split('\n')[:-1]
        tmp = tmp[0].split(',')
        pred = tmp[-2].lower()
        gt = tmp[-1].lower()
        y_true.append(C[gt])
        y_pred.append(C[pred])
    
    print(len(y_true), len(y_pred))
    
    cnf_matrix = confusion_matrix(y_true, y_pred)

    import warnings
    warnings.filterwarnings('ignore')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix_new(cnf_matrix, classes=C.keys(), normalize=True,
                          run=run,what=what)

    result = calculateRecallPrecisionAndF1Score(y_true, y_pred)
    print(result)
    # print("==== F1-Score ====")
    # print(f1_score(y_true, y_pred, average='micro'))
    # print(f1_score(y_true, y_pred, average='macro'))
    # print(f1_score(y_true, y_pred, average='weighted'))
            
    # print("==== recall_score ====")
    # print(recall_score(y_true, y_pred, average='micro'))
    # print(recall_score(y_true, y_pred, average='macro'))
    # print(recall_score(y_true, y_pred, average='weighted'))

    # print("==== F1-Score ====")
    # print(precision_score(y_true, y_pred, average='micro'))
    # print(precision_score(y_true, y_pred, average='macro'))
    # print(precision_score(y_true, y_pred, average='weighted'))
    
    return PredictedandGroundTruthList


def RunTestAndGenerateConWithModelAndWeightFile(modelFile, amountofTestImages, datapathTestImages, C, Classes, run, what):
    PredictedandGroundTruthList = []

    test_datagen = ImageDataGenerator(rescale = 1./255)

    test_set = test_datagen.flow_from_directory(datapathTestImages,
                                                target_size = (256, 256),
                                                batch_size = amountofTestImages,
                                                color_mode = "grayscale")


    model = load_model(modelFile)
    #model = load_model('CNN_Model_LowRes_RailType10Epochs_8000StepsPreEpochs.h5')


    for i in range(amountofTestImages):
        image, label = test_set._get_batches_of_transformed_samples(np.array([i]))
        #image = test_set._get_batches_of_transformed_samples(np.array([i]))
        image_name = test_set.filenames[i]
        pred = model.predict_classes(image)
        tmp = image_name.split('\\')[-1]
        if label[0][0] == 1:
            PredictedandGroundTruthList.append(str(tmp[:-4]) + ',' + Classes[pred[0]] + "," + Classes[0] + "\n")
        elif label[0][1] == 1:
            PredictedandGroundTruthList.append(str(tmp[:-4]) + ',' + Classes[pred[0]] + "," + Classes[1] + "\n")
        elif label[0][2] == 1:
            PredictedandGroundTruthList.append(str(tmp[:-4]) + ',' + Classes[pred[0]] + "," + Classes[2] + "\n") 
    
    y_true = []
    y_pred = []
    
    for i in range(len(PredictedandGroundTruthList)):
        tmp = PredictedandGroundTruthList[i].split('\n')[:-1]
        tmp = tmp[0].split(',')
        pred = tmp[-2].lower()
        gt = tmp[-1].lower()
        y_true.append(C[gt])
        y_pred.append(C[pred])
    
    print(len(y_true), len(y_pred))
    
    cnf_matrix = confusion_matrix(y_true, y_pred)

    import warnings
    warnings.filterwarnings('ignore')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix_new(cnf_matrix, classes=C.keys(), normalize=True,
                          run=run,what=what)

    result = calculateRecallPrecisionAndF1Score(y_true, y_pred)
    print(result)
    # print("==== F1-Score ====")
    # print(f1_score(y_true, y_pred, average='micro'))
    # print(f1_score(y_true, y_pred, average='macro'))
    # print(f1_score(y_true, y_pred, average='weighted'))
            
    # print("==== recall_score ====")
    # print(recall_score(y_true, y_pred, average='micro'))
    # print(recall_score(y_true, y_pred, average='macro'))
    # print(recall_score(y_true, y_pred, average='weighted'))

    # print("==== F1-Score ====")
    # print(precision_score(y_true, y_pred, average='micro'))
    # print(precision_score(y_true, y_pred, average='macro'))
    # print(precision_score(y_true, y_pred, average='weighted'))
    
    return PredictedandGroundTruthList







