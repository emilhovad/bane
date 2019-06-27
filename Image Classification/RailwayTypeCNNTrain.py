# Convolutional Neural Network


from numpy.random import seed
seed(7)
#from tensorflow import set_random_seed
#set_random_seed(7)
import os
import keras
import tensorflow as tf
tf.compat.v1.set_random_seed(7)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU, Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, EarlyStopping, Callback
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from keras.constraints import maxnorm
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

#from collections import Counter
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", required=True,
	# help="path to output plot")
ap.add_argument("-g", "--gpus", type=int, default=1,
	help="# of GPUs to use for training")
args = vars(ap.parse_args())

G = args["gpus"]
datapath = '../../dataset/FirstSplitToTrainValiAndTest/TrainSetRailWayAugmented'
datapathVal = '../../dataset/FirstSplitToTrainValiAndTest/ValidationSet'
MODELOUTPUT = 'cnnHighAllDataBaseline'
ModelPath = 'models/' + MODELOUTPUT + '/'
CSV_FILE_DISTINATION =  MODELOUTPUT + '.csv'
IMG_SIZE = 256
BACH_SIZE = 64 
NUM_EPOCHS = 80 #40
INIT_LR = 0.001#5e-3
#CNN Baseline training is ModelNumber = 1    
#CNN improved training is ModelNumber = 2
ModelNumber = 1 

def Create_Dir(path):
    try:  
        os.makedirs(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s" % path)

def lr_scheduler(epoch, lr):
    EpochValues = [5,10,20]
    decay_rate = 0.5
    for EpochV in EpochValues:
        if epoch == EpochV:
            return lr * decay_rate
    return lr
    
def poly_decay(epoch):
	# initialize the maximum number of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = NUM_EPOCHS
	baseLR = INIT_LR
	power = 1.0
 
	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
 
	# return the new learning rate
	return alpha

class CustomModelCheckpoint(Callback):

    def __init__(self, model, path):

        super().__init__()

        # This is the argument that will be modify by fit_generator
        # self.model = model
        self.path = path

        # We set the model (non multi gpu) under an other name
        self.model_for_saving = model
        
        self.best_acc = 0
        self.start_time = time.monotonic()

    def on_epoch_end(self, epoch, logs=None):

        acc = logs['val_acc']
        oldTime = self.start_time
        self.start_time = time.monotonic()
        elapsed_time = self.start_time - oldTime
        
        # Here we save the original one
        
        print('Time Elapsed: ' + time.strftime("%M:%S", time.gmtime(elapsed_time)))
        print("\nSaving model to : {}".format(self.path.format(epoch=epoch, val_acc=acc)))
        self.model_for_saving.save_weights(self.path.format(epoch=epoch, val_acc=acc), overwrite=True)
        self.best_acc = acc
        
        if acc > self.best_acc:
            print("_____________________________________________________________________________")
            print("Model have better ACC")
            print("_____________________________________________________________________________")
            # print('Time Elapsed: ' + time.strftime("%M:%S", time.gmtime(elapsed_time)))
            # print("\nSaving model to : {}".format(self.path.format(epoch=epoch, val_acc=acc)))
            # self.model_for_saving.save_weights(self.path.format(epoch=epoch, val_acc=acc), overwrite=True)
            # self.best_acc = acc

def buildModelBaseline(IMG_SIZE, L2, dropoutVal):
    # Initialising the CNN
    model = Sequential()

    # Step 1 - Convolution
    #model.add(Conv2D(16, (1, 1), input_shape = (IMG_SIZE, IMG_SIZE, 1), activation = 'relu', kernel_regularizer=l2(L2)))
    #model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_regularizer=l2(L2)))
    model.add(Conv2D(32, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 1), activation = 'relu', kernel_regularizer=l2(L2)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_regularizer=l2(L2)))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
    model.add(Flatten())

    # Step 4 - Full connection
    #model.add(Dense(units = 2048, activation = 'relu'))
    model.add(Dropout(dropoutVal))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(dropoutVal))
    model.add(Dense(units = 3, activation = 'softmax'))


    for layer in model.layers:
        print(layer.input_shape)
        print(layer.output_shape)

    return model

def buildModel2(IMG_SIZE, L2, dropoutVal):
    # Initialising the CNN
    model = Sequential()

    # Step 1 - Convolution
    model.add(Conv2D(32, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 1), activation = 'relu', kernel_regularizer=l2(L2)))
    model.add(Conv2D(64, (3, 3), strides=2, activation = 'relu', kernel_regularizer=l2(L2)))
    model.add(Conv2D(32, (1, 1), activation = 'relu', kernel_regularizer=l2(L2)))
    model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_regularizer=l2(L2)))

    # Step 2 - Pooling
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
    model.add(Conv2D(128, (3, 3), strides=2, activation = 'relu', kernel_regularizer=l2(L2)))
    model.add(Conv2D(64, (1, 1), activation = 'relu', kernel_regularizer=l2(L2)))
    model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_regularizer=l2(L2)))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
    model.add(Flatten())

    # Step 4 - Full connection
    model.add(Dropout(dropoutVal))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(dropoutVal))
    model.add(Dense(units = 3, activation = 'softmax'))
    
    
    for layer in model.layers:
        #print(layer.input_shape)
        print(layer.output_shape)

    return model
    
def buildModel3(IMG_SIZE, L2, dropoutVal):
    # Initialising the CNN
    model = Sequential()

    # Step 1 - Convolution
    model.add(Conv2D(32, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 1), activation = 'relu', kernel_regularizer=l2(L2)))

    # Step 2 - Pooling
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
    model.add(Conv2D(16, (1, 1), activation = 'relu', kernel_regularizer=l2(L2)))
    model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_regularizer=l2(L2)))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
    model.add(Flatten())

    # Step 4 - Full connection
    #model.add(Dense(units = 1024, activation = 'relu'))
    model.add(Dropout(dropoutVal))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(dropoutVal))
    model.add(Dense(units = 3, activation = 'softmax'))
    
    
    # for layer in model.layers:
        # print(layer.input_shape)
        # print(layer.output_shape)

    return model
    
def buildModel4(IMG_SIZE, L2, dropoutVal):
    # Initialising the CNN
    model = Sequential()

    # Step 1 - Convolution
    #model.add(Conv2D(16, (1, 1), input_shape = (IMG_SIZE, IMG_SIZE, 1), activation = 'relu', kernel_regularizer=l2(L2)))
    model.add(Conv2D(32, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 1), activation = 'relu', padding='same', kernel_regularizer=l2(L2)))

    # Step 2 - Pooling
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding='same', kernel_regularizer=l2(L2)))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a third convolutional layer
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding='same', kernel_regularizer=l2(L2)))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
    model.add(Flatten())

    # Step 4 - Full connection
    model.add(Dropout(dropoutVal))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(dropoutVal))
    model.add(Dense(units = 3, activation = 'softmax'))
    
    
    for layer in model.layers:
        print(layer.input_shape)
        print(layer.output_shape)

    return model

def buildModel5(IMG_SIZE, L2, dropoutVal):
    # Initialising the CNN
    model = Sequential()

    # Step 1 - Convolution
    #model.add(Conv2D(16, (1, 1), input_shape = (IMG_SIZE, IMG_SIZE, 1), activation = 'relu', kernel_regularizer=l2(L2)))
    model.add(Conv2D(32, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 1), activation = 'relu', padding='same', kernel_regularizer=l2(L2)))

    # Step 2 - Pooling
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding='same', kernel_regularizer=l2(L2)))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a third convolutional layer
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding='same', kernel_regularizer=l2(L2)))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
    model.add(Flatten())

    # Step 4 - Full connection
    model.add(Dropout(dropoutVal))
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dropout(dropoutVal))
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dropout(dropoutVal))
    model.add(Dense(units = 3, activation = 'softmax'))
    
    
    for layer in model.layers:
        print(layer.input_shape)
        print(layer.output_shape)

    return model
    
def buildModel6(IMG_SIZE, L2, dropoutVal):
    # Initialising the CNN
    model = Sequential()

    # Step 1 - Convolution
    #model.add(Conv2D(16, (1, 1), input_shape = (IMG_SIZE, IMG_SIZE, 1), activation = 'relu', kernel_regularizer=l2(L2)))
    model.add(Conv2D(32, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 1), activation = 'relu', padding='same', kernel_regularizer=l2(L2)))

    # Step 2 - Pooling
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding='same', kernel_regularizer=l2(L2)))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a third convolutional layer
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding='same', kernel_regularizer=l2(L2)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a fourt convolutional layer
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding='same', kernel_regularizer=l2(L2)))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
    model.add(Flatten())

    # Step 4 - Full connection
    model.add(Dropout(dropoutVal))
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dropout(dropoutVal))
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dropout(dropoutVal))
    model.add(Dense(units = 3, activation = 'softmax'))
    
    
    for layer in model.layers:
        print(layer.input_shape)
        print(layer.output_shape)

    return model

def CNNBaseline(IMG_SIZE, INIT_LR, L2, dropoutVal, optimizer, datapath, datapathVal, ModelPath, WeightsPath, CSV_FILE_DISTINATION, ModelToRun):

    model = Sequential()
    print("_____________________________________________________________________________")
    print("Model To run: " + str(ModelToRun))
    print("_____________________________________________________________________________")
    if G <= 1:
        print("[INFO] training with 1 GPU...")
        #model = buildModel(IMG_SIZE)
        if ModelToRun == 1:
            print("_____________________________________________________________________________")
            print("Bach size to use: " + str(BACH_SIZE))
            print("_____________________________________________________________________________")
            print(BACH_SIZE)
            model = buildModelBaseline(IMG_SIZE, L2, dropoutVal)
        elif ModelToRun== 2:
            model = buildModel2(IMG_SIZE, L2, dropoutVal)
        elif ModelToRun== 3:
            model = buildModel3(IMG_SIZE, L2, dropoutVal)
        elif ModelToRun== 4:
            model = buildModel4(IMG_SIZE, L2, dropoutVal)
        elif ModelToRun== 5:
            model = buildModel5(IMG_SIZE, L2, dropoutVal)
        elif ModelToRun== 6:
            model = buildModel6(IMG_SIZE, L2, dropoutVal)
        gpu_model = model
    else:
        print("[INFO] training with {} GPUs...".format(G))

        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
            # initialize the model
            #model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
            #model = buildModel(IMG_SIZE)
            if ModelToRun == 1:
                model = buildModelBaseline(IMG_SIZE, L2, dropoutVal)
            elif ModelToRun== 2:
                model = buildModel2(IMG_SIZE, L2, dropoutVal)
            elif ModelToRun== 3:
                model = buildModel3(IMG_SIZE, L2, dropoutVal)
            elif ModelToRun== 4:
                model = buildModel4(IMG_SIZE, L2, dropoutVal)
            elif ModelToRun== 5:
                model = buildModel5(IMG_SIZE, L2, dropoutVal)
            elif ModelToRun== 6:
                model = buildModel6(IMG_SIZE, L2, dropoutVal)
        
        # make the model parallel
        gpu_model = multi_gpu_model(model, gpus=G)

    # Compiling the CNN
    #optimizerToUse = optimizers.Adam(lr=INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    optimizerToUse = optimizers.Adam(lr=INIT_LR, epsilon=None, decay=1e-6, amsgrad=False)
    gpu_model.compile(optimizer = optimizerToUse, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # Part 2 - Fitting the CNN to the images
    saveName = ModelPath + MODELOUTPUT + '.h5'
    model.save(saveName) 


    train_datagen = ImageDataGenerator(rescale = 1./255,
                                            #shear_range = 0.2,
                                            #zoom_range = 0.2,
                                            #horizontal_flip = True,
                                            #vertical_flip=True,
                                            brightness_range=[1.0, 1.5])

    Validation_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(datapath,
                                                     target_size = (IMG_SIZE, IMG_SIZE),
                                                     batch_size = BACH_SIZE,
                                                     color_mode = "grayscale",
                                                     shuffle=True,
                                                     seed=42)
                                                     #,class_mode = 'binary')

    validation_set = Validation_datagen.flow_from_directory(datapathVal,
                                                     target_size = (IMG_SIZE, IMG_SIZE),
                                                     batch_size = BACH_SIZE,
                                                     color_mode = "grayscale",
                                                     shuffle=True,
                                                     seed=42)
                     
    presetSteps_pre_epochs = training_set.n // (BACH_SIZE * G)
    validation_steps = validation_set.n // (BACH_SIZE * G)

    # counter = Counter(training_set.classes)                          
    # max_val = float(max(counter.values()))       
    # class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}  

    class_weights = class_weight.compute_class_weight(
                   'balanced',
                    np.unique(training_set.classes), 
                    training_set.classes)

    #model = load_model('CNN_Model_HighRes_RailType4Epochs_8000StepsPreEpochs.h5')
    filepath = WeightsPath + 'CNN_weights_epoch_{epoch:02d}_val_acc_{val_acc:.3f}_Steps_' + str(presetSteps_pre_epochs) + '.h5'
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    checkpoint = CustomModelCheckpoint(model, filepath)
    checkpoint2 = LearningRateScheduler(lr_scheduler, verbose=1)
    checkpoint3 = CSVLogger(CSV_FILE_DISTINATION, separator=',', append=True)
    checkpoint4 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    #history = LossHistory()
    #callbacks_list = [checkpoint, checkpoint3]
    callbacks_list = [checkpoint, checkpoint3, checkpoint4]
    #callbacks_list = [checkpoint, checkpoint2, checkpoint3, history]
                            
    Hist = gpu_model.fit_generator(training_set,
                             steps_per_epoch = presetSteps_pre_epochs,
                             epochs = NUM_EPOCHS,
                             validation_data = validation_set,
                             validation_steps = validation_steps,
                             max_queue_size=100,
                             workers=40,
                             callbacks=callbacks_list, 
                             class_weight=class_weights)

    return Hist
    # saveName = 'CNN_Model_HighRes_RailType' + str(presetEpochs) + 'Epochs_' + str(presetSteps_pre_epochs) + 'StepsPreEpochs.h5'
    # classifier.save(saveName) 
    
def PlotHistory(Hist, ModelPath, MODELOUTPUT, STEPONEOUTPUT):
    yMin = 0.0
    yMax = 0.8
    # grab the history object dictionary
    Hist = Hist.history
     
    # plot the training loss and accuracy
    N = np.arange(0, len(Hist["loss"]))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, Hist["loss"], label="train_loss")
    plt.plot(N, Hist["val_loss"], label="Validation_loss")
    plt.title('CNN_' + STEPONEOUTPUT + '_Loss')
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.ylim(yMin,yMax)
    plt.xlim(0,40)
     
    # save the figure
    plt.savefig(ModelPath + MODELOUTPUT + '_loss')
    plt.close()

    yMin = 0.7
    yMax = 1.0
    # plot the training loss and accuracy
    N = np.arange(0, len(Hist["loss"]))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, Hist["acc"], label="train_acc")
    plt.plot(N, Hist["val_acc"], label="Validation_acc")
    plt.title('CNN_' + STEPONEOUTPUT + '_acc')
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.ylim(yMin,yMax)
    plt.xlim(0,40)
    # save the figure
    plt.savefig(ModelPath + MODELOUTPUT + '_acc')
    plt.close()
    
def PlotHistoryTotal(Hist, ModelPath, MODELOUTPUT, STEPONEOUTPUT):
    yMin = 0.0
    yMax = 1.0
    # grab the history object dictionary
    Hist = Hist.history
     
    # plot the training loss and accuracy
    N = np.arange(0, len(Hist["loss"]))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, Hist["loss"], label="train_loss")
    plt.plot(N, Hist["val_loss"], label="Validation_loss")
    plt.plot(N, Hist["acc"], label="train_acc")
    plt.plot(N, Hist["val_acc"], label="Validation_acc")
    plt.title('CNN' + STEPONEOUTPUT)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.ylim(yMin,yMax)
     
    # save the figure
    plt.savefig(ModelPath + MODELOUTPUT + '_loss')
    plt.close()

def TrainMultipleModels(ModelToUse, NumbersIfModels):
    optimizer = 'Adam'
    INIT_LR = 0.0015 #Bedst 0.0005
    dropoutVal = 0.55 #Bedst 0.30
    L2 = 0.00005 #Bedst 0.0001

    for NumbersIfModel in range(NumbersIfModels):
        tf.reset_default_graph() # for being sure
        keras.backend.clear_session()
        STEPONEOUTPUT = 'LR_' + str(INIT_LR) + '_DP_' + str(dropoutVal) + '_L2_' + str(L2) + '_Model_' + str(NumbersIfModel+1)
        STEPONEOUTPUT = STEPONEOUTPUT.replace(".","")
        MODELOUTPUT = 'CNN_'+ str(STEPONEOUTPUT) +'_AllDataAugmented'
        ModelPath = 'ModelsFinal/CNNModelUsed_' + str(ModelToUse) + '/' + MODELOUTPUT + '/'
        WeightsPath = ModelPath + '/weights/'
        CSV_FILE_DISTINATION =  ModelPath + MODELOUTPUT + '.csv'
    
        Create_Dir(ModelPath)
        Create_Dir(WeightsPath)

        HIST = CNNBaseline(IMG_SIZE, INIT_LR, L2, dropoutVal, optimizer, datapath, datapathVal, ModelPath, WeightsPath, CSV_FILE_DISTINATION, ModelToUse)
        STEPONEOUTPUT = ""
        PlotHistoryTotal(HIST, ModelPath, MODELOUTPUT, STEPONEOUTPUT)
        

def TrainMultipleModelsBaseline(ModelToUse, NumbersIfModels):
    optimizer = 'Adam'
    INIT_LR = 0.0015 #Bedst 0.0005
    dropoutVal = 0.55 #Bedst 0.30
    L2 = 0.00005 #Bedst 0.0001
    
    for NumbersIfModel in range(NumbersIfModels):
        seed = 7
        np.random.seed(seed)
        STEPONEOUTPUT = 'LR_' + str(INIT_LR) + '_DP_' + str(dropoutVal) + '_L2_' + str(L2) + '_Model_' + str(NumbersIfModel)
        STEPONEOUTPUT = STEPONEOUTPUT.replace(".","")
        #MODELOUTPUT = 'CNNBaseline4_'+ str(STEPONEOUTPUT) +'_AllDataAugmented'
        ModelPath = 'ModelsFinal/CNNModelUsed_' + str(ModelToUse) + '/' + MODELOUTPUT + '/'
        WeightsPath = ModelPath + '/weights/'
        CSV_FILE_DISTINATION =  ModelPath + MODELOUTPUT + '.csv'
    
        Create_Dir(ModelPath)
        Create_Dir(WeightsPath)

        HIST = CNNBaseline(IMG_SIZE, INIT_LR, L2, dropoutVal, optimizer, datapath, datapathVal, ModelPath, WeightsPath, CSV_FILE_DISTINATION, ModelToUse)

        PlotHistory(HIST, ModelPath, MODELOUTPUT, STEPONEOUTPUT)
        tf.reset_default_graph() # for being sure
        keras.backend.clear_session()
		
def TrainTestDropoutValues(dropoutVals):
    optimizer = 'Adam'
    INIT_LR = 0.00075 #Bedst 0.0005
    #dropoutVal = 0.35 #Bedst 0.35
    #dropoutVals = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    L2 = 0.0001 #Bedst 0.0001
    
    #for NumbersIfModel in range(NumbersIfModels):
    NumbersIfModel = 1
    for dropoutVal in dropoutVals:
        seed = 7
        np.random.seed(seed)
        set_random_seed(seed)
        tf.reset_default_graph() # for being sure
        keras.backend.clear_session()
        STEPONEOUTPUT = 'ModelNR_' + str(NumbersIfModel) + '_LR_' + str(INIT_LR) + '_DP_' + str(dropoutVal) + '_L2_' + str(L2)
        STEPONEOUTPUT = STEPONEOUTPUT.replace(".","")
        #MODELOUTPUT = 'CNNBaseline4_'+ str(STEPONEOUTPUT) +'_AllDataAugmented'
        MODELOUTPUT = 'CNN_'+ str(STEPONEOUTPUT) +'_TrainDataAugmented'
        ModelPath = 'Models/CNNTestDropout/' + MODELOUTPUT + '/'
        WeightsPath = ModelPath + '/weights/'
        CSV_FILE_DISTINATION =  ModelPath + MODELOUTPUT + '.csv'
    
        Create_Dir(ModelPath)
        Create_Dir(WeightsPath)

        HIST = CNNBaseline(IMG_SIZE, INIT_LR, L2, dropoutVal, optimizer, datapath, datapathVal, ModelPath, WeightsPath, CSV_FILE_DISTINATION, False)
        NumbersIfModel += 1

        PlotHistory(HIST, ModelPath, MODELOUTPUT, STEPONEOUTPUT)
        

def TrainTestLearningRateValues(INIT_LRs):
    optimizer = 'Adam'
    #INIT_LR = 0.0005 #Bedst 0.0005
    dropoutVal = 0.35 #Bedst 0.35
	#dropoutVals = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    L2 = 0.0001 #Bedst 0.0001
    
    #for NumbersIfModel in range(NumbersIfModels):
    NumbersIfModel = 1
    for INIT_LR in INIT_LRs:
        seed = 7
        np.random.seed(seed)
        set_random_seed(seed)
        tf.reset_default_graph() # for being sure
        keras.backend.clear_session()
        STEPONEOUTPUT = 'ModelNR_' + str(NumbersIfModel) + '_LR_' + str(INIT_LR) + '_DP_' + str(dropoutVal) + '_L2_' + str(L2)
        STEPONEOUTPUT = STEPONEOUTPUT.replace(".","")
        #MODELOUTPUT = 'CNNBaseline4_'+ str(STEPONEOUTPUT) +'_AllDataAugmented'
        MODELOUTPUT = 'CNN_'+ str(STEPONEOUTPUT) +'_TrainDataAugmented'
        ModelPath = 'Models/CNNTestLearningRate/' + MODELOUTPUT + '/'
        WeightsPath = ModelPath + '/weights/'
        CSV_FILE_DISTINATION =  ModelPath + MODELOUTPUT + '.csv'
    
        Create_Dir(ModelPath)
        Create_Dir(WeightsPath)

        HIST = CNNBaseline(IMG_SIZE, INIT_LR, L2, dropoutVal, optimizer, datapath, datapathVal, ModelPath, WeightsPath, CSV_FILE_DISTINATION, False)
        NumbersIfModel += 1

        PlotHistory(HIST, ModelPath, MODELOUTPUT, STEPONEOUTPUT) 

def TrainTestOptimizerValues(optimizers):
    #optimizer = 'Adam'
    INIT_LR = 0.0005 #Bedst 0.0005
    dropoutVal = 0.35 #Bedst 0.35
	#dropoutVals = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    L2 = 0.0001 #Bedst 0.0001
    
    #for NumbersIfModel in range(NumbersIfModels):
    NumbersIfModel = 1
    for optimizer in optimizers:
        seed = 7
        np.random.seed(seed)
        set_random_seed(seed)
        tf.reset_default_graph() # for being sure
        keras.backend.clear_session()
        STEPONEOUTPUT = '_Op_' + str(optimizer) + '_LR_' + str(INIT_LR) + '_DP_' + str(dropoutVal) + '_L2_' + str(L2)
        STEPONEOUTPUT = STEPONEOUTPUT.replace(".","")
        #MODELOUTPUT = 'CNNBaseline4_'+ str(STEPONEOUTPUT) +'_AllDataAugmented'
        MODELOUTPUT = 'CNN_'+ str(STEPONEOUTPUT) +'_TrainDataAugmented'
        ModelPath = 'Models/CNNTestOptimizer/' + MODELOUTPUT + '/'
        WeightsPath = ModelPath + '/weights/'
        CSV_FILE_DISTINATION =  ModelPath + MODELOUTPUT + '.csv'
    
        Create_Dir(ModelPath)
        Create_Dir(WeightsPath)

        HIST = CNNBaseline(IMG_SIZE, INIT_LR, L2, dropoutVal, optimizer, datapath, datapathVal, ModelPath, WeightsPath, CSV_FILE_DISTINATION, False)
        NumbersIfModel += 1

        PlotHistory(HIST, ModelPath, MODELOUTPUT, STEPONEOUTPUT)

def TrainTestRegularizationValues(L2s):
    optimizer = 'Adam'
    INIT_LR = 0.00075 #Bedst 0.0005
    dropoutVal = 0.5 #Bedst 0.35
	#dropoutVals = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    #L2 = 0.0001 #Bedst 0.0001
    
    #for NumbersIfModel in range(NumbersIfModels):
    NumbersIfModel = 1
    for L2 in L2s:
        seed = 7
        np.random.seed(seed)
        set_random_seed(seed)
        tf.reset_default_graph() # for being sure
        keras.backend.clear_session()
        STEPONEOUTPUT = 'ModelNR_' + str(NumbersIfModel) + '_LR_' + str(INIT_LR) + '_DP_' + str(dropoutVal) + '_L2_' + str(L2)
        STEPONEOUTPUT = STEPONEOUTPUT.replace(".","")
        #MODELOUTPUT = 'CNNBaseline4_'+ str(STEPONEOUTPUT) +'_AllDataAugmented'
        MODELOUTPUT = 'CNN_'+ str(STEPONEOUTPUT) +'_TrainDataAugmented'
        ModelPath = 'Models/CNNTestRegularization/' + MODELOUTPUT + '/'
        WeightsPath = ModelPath + '/weights/'
        CSV_FILE_DISTINATION =  ModelPath + MODELOUTPUT + '.csv'
    
        Create_Dir(ModelPath)
        Create_Dir(WeightsPath)

        HIST = CNNBaseline(IMG_SIZE, INIT_LR, L2, dropoutVal, optimizer, datapath, datapathVal, ModelPath, WeightsPath, CSV_FILE_DISTINATION, False)
        NumbersIfModel += 1

        PlotHistory(HIST, ModelPath, MODELOUTPUT, STEPONEOUTPUT) 
# cudaTest = tf.test.is_gpu_available(
    # cuda_only=False,
    # min_cuda_compute_capability=None
# )
# print(cudaTest)
#TrainTestOptimizerValues(["Adadelta","Adagrad","Adamax","Adam","Nadam","RMSprop","SGD"])
#TrainTestLearningRateValues([0.0001,0.00025,0.0005,0.00075,0.001,0.00125,0.0015,0.00175,0.002])
#TrainTestDropoutValues([0.1,0.125,0.15,0.175,0.2,0.225,0.250,0.275,0.3,0.325,0.350,0.375,0.4,0.425,0.450,0.475,0.5,0.525,0.550,0.575,0.6])
#TrainTestRegularizationValues([0.000025,0.00005,0.000075,0.0001,0.000125,0.00015,0.000175,0.0002,0.000225,0.00025,0.000275,0.0003,0.000325,0.00035,0.000375,0.0004,0.000425,0.00045,0.000375,0.0005])

#TrainMultipleModels(2,5)
TrainMultipleModels(ModelNumber,1)
#TrainMultipleModels(4,5)
#TrainMultipleModels(5,5)

#TrainMultipleModelsBaseline(1)

#TrainMultipleModelsBaseline(5)
#TrainMultipleModels(5)
#HIST = CNNBaseline(IMG_SIZE, datapath, datapathVal, ModelPath, CSV_FILE_DISTINATION)
#numpy_loss_history = numpy.array(HIST)
#numpy.savetxt(MODELOUTPUT + "loss_history.txt", numpy_loss_history, delimiter=",")
#PlotHistory(HIST, MODELOUTPUT)












                  
