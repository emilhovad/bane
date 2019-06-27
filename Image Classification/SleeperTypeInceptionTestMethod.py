# test.py

import os
import tensorflow as tf
import numpy as np
import cv2
import shutil
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from plot_confusion_matrix import plot_confusion_matrix_new
import matplotlib.pyplot as plt

# module-level variables ##############################################################################################
RETRAINED_LABELS_TXT_FILE_LOC = os.getcwd() + "../Training/Inceotionv3/" + "retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC = os.getcwd() + "../Training/Inceotionv3/" + "retrained_graph.pb"

TEST_IMAGES_DIR = os.getcwd() + "/testSetBTR_24-29"
FILE_DISTINATION = './result/testSet2/'
SPLIDT_CONFIDENCE = 0.9

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)

CSV_FILE_DISTINATION = './result/testSetPredict24-29.txt'
#OutputCSVFile = False
OutputCSVFile = True
DoNotPrint = True

#######################################################################################################################

def main(RETRAINED_LABELS, RETRAINED_GRAPH, TEST_IMAGES, groundTruthtxt, C, Classes, run, what):
    print('test_images: ' + TEST_IMAGES)
    if not len(RETRAINED_LABELS) == 0:
        RETRAINED_LABELS_TXT_FILE_LOC = RETRAINED_LABELS
        RETRAINED_GRAPH_PB_FILE_LOC = RETRAINED_GRAPH
        TEST_IMAGES_DIR = TEST_IMAGES
            
    print(TEST_IMAGES_DIR)
    print("starting program . . .")
    FILE_DISTINATION = './result/testSet2/'
    if not checkIfNecessaryPathsAndFilesExist(TEST_IMAGES_DIR, RETRAINED_LABELS_TXT_FILE_LOC, RETRAINED_GRAPH_PB_FILE_LOC):
        return
    # end if

    predictionsList = []
    
    # get a list of classifications from the labels file
    classifications = []
    # for each line in the label file . . .
    for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
        # remove the carriage return
        classification = currentLine.rstrip()
        # and append to the list
        classifications.append(FILE_DISTINATION + classification)
    # end for
    
    #Folders = FILE_DISTINATION.split('/')
    #Create_Dir(Folders[-2])
    #Create_Dir(Folders[-2] + '/'+ Folders[-1])
    
    # for each element in classifications
    if not OutputCSVFile:
        for classification in classifications:
            if classification.endswith("s"):
                classification = classification[:-1]
            classification = classification.split('/')[-1]
            #create folder if none exist
            Create_Dir(FILE_DISTINATION + 'newCorrecting/' + classification)
            Create_Dir(FILE_DISTINATION + 'Correcting/' + classification)
            Create_Dir(FILE_DISTINATION + classification)

    # show the classifications to prove out that we were able to read the label file successfully
    print("classifications = " + str(classifications))

    # load the graph from file
    with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
        # instantiate a GraphDef object
        graphDef = tf.GraphDef()
        # read in retrained graph into the GraphDef object
        graphDef.ParseFromString(retrainedGraphFile.read())
        # import the graph into the current default Graph, note that we don't need to be concerned with the return value
        _ = tf.import_graph_def(graphDef, name='')
    # end with

    # if the test image directory listed above is not valid, show an error message and bail
    if not os.path.isdir(TEST_IMAGES_DIR):
        print("the test image directory does not seem to be a valid directory, check file / directory paths")
        return
    # end if

    with tf.Session() as sess:
        
        # for each file in the test images directory . . .
        for fileName in os.listdir(TEST_IMAGES_DIR):
            # if the file does not end in .jpg or .jpeg (case-insensitive), continue with the next iteration of the for loop
            if not (fileName.lower().endswith(".jpg") or fileName.lower().endswith(".jpeg")):
                continue
            # end if

            # show the file name on std out
            if not DoNotPrint:
                print(fileName)

            # get the file name and full path of the current image file
            imageFileWithPath = os.path.join(TEST_IMAGES_DIR, fileName)
            # attempt to open the image with OpenCV
            openCVImage = cv2.imread(imageFileWithPath)

            # if we were not able to successfully open the image, continue with the next iteration of the for loop
            if openCVImage is None:
                print("unable to open " + fileName + " as an OpenCV image")
                continue
            # end if

            # get the final tensor from the graph
            finalTensor = sess.graph.get_tensor_by_name('final_result:0')

            # convert the OpenCV image (numpy array) to a TensorFlow image
            tfImage = np.array(openCVImage)[:, :, 0:3]
            
            # run the network to get the predictions
            predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

            # sort predictions from most confidence to least confidence
            sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

            if not DoNotPrint:
                print("---------------------------------------")

            # keep track of if we're going through the next for loop for the first time so we can show more info about
            # the first prediction, which is the most likely prediction (they were sorted descending above)
            onMostLikelyPrediction = True
            # for each prediction . . .
            for prediction in sortedPredictions:
                strClassification = classifications[prediction]

                # if the classification (obtained from the directory name) ends with the letter "s", remove the "s" to change from plural to singular
                if strClassification.endswith("s"):
                    strClassification = strClassification[:-1]
                # end if
                strClassification = strClassification.split('/')[-1]

                # get confidence, then get confidence rounded to 2 places after the decimal
                confidence = predictions[0][prediction]

                # if we're on the first (most likely) prediction, state what the object appears to be and show a % confidence to two decimal places
                if onMostLikelyPrediction:
                    # get the score as a %
                    scoreAsAPercent = confidence * 100.0
                    # show the result to std out
                    if not DoNotPrint:
                        print("the object appears to be a " + strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                    # write the result on the image
                    tmp = imageFileWithPath.split('/')[-1]
                    #print(tmp)
                    tmp = tmp.split('\\')[-1]
                    #print(tmp)
                    #print('/result/' + strClassification)
                    #print(confidence)
                    if OutputCSVFile:
                        #print('i am here')
                        predictionsList.append(tmp[:-4] + ',' + str(confidence) + ',' + strClassification)
                        # with open(CSV_FILE_DISTINATION, 'a') as f:
                            # f.write(tmp[:-4] + ',' + strClassification + "\n")
                    else:
                        if confidence < SPLIDT_CONFIDENCE:
                            if not DoNotPrint:
                                print(FILE_DISTINATION + 'newCorrecting/'+ strClassification)
                            shutil.copy(tmp, FILE_DISTINATION + 'newCorrecting/'+ strClassification)
                        else:
                            if not DoNotPrint:
                                print(FILE_DISTINATION + strClassification)
                            shutil.copy(tmp, FILE_DISTINATION + strClassification)
                    #writeResultOnImage(openCVImage, strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                    # finally we can show the OpenCV image
                    #cv2.imshow(fileName, openCVImage)
                    # mark that we've show the most likely prediction at this point so the additional information in
                    # this if statement does not show again for this image
                    onMostLikelyPrediction = False
                # end if

                # for any prediction, show the confidence as a ratio to five decimal places
                if not DoNotPrint:
                    print(strClassification + " (" +  "{0:.5f}".format(confidence) + ")")
            # end for

            # pause until a key is pressed so the user can see the current image (shown above) and the prediction info
            cv2.waitKey()
            # after a key is pressed, close the current window to prep for the next time around
            cv2.destroyAllWindows()
        # end for
    # end with

    # write the graph to file so we can view with TensorBoard
    tfFileWriter = tf.summary.FileWriter(os.getcwd())
    tfFileWriter.add_graph(sess.graph)
    tfFileWriter.close()
    
    createConfusionMatrix(predictionsList, groundTruthtxt, C, run, what)
    return predictionsList

# end main

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist(TEST_IMAGES_DIR, RETRAINED_LABELS_TXT_FILE_LOC, RETRAINED_GRAPH_PB_FILE_LOC):
    if not os.path.exists(TEST_IMAGES_DIR):
        print('')
        print('ERROR: TEST_IMAGES_DIR "' + TEST_IMAGES_DIR + '" does not seem to exist')
        print('Did you set up the test images?')
        print('')
        return False
    # end if

    if not os.path.exists(RETRAINED_LABELS_TXT_FILE_LOC):
        print('ERROR: RETRAINED_LABELS_TXT_FILE_LOC "' + RETRAINED_LABELS_TXT_FILE_LOC + '" does not seem to exist')
        return False
    # end if

    if not os.path.exists(RETRAINED_GRAPH_PB_FILE_LOC):
        print('ERROR: RETRAINED_GRAPH_PB_FILE_LOC "' + RETRAINED_GRAPH_PB_FILE_LOC + '" does not seem to exist')
        return False
    # end if

    return True
# end function

#######################################################################################################################
def writeResultOnImage(openCVImage, resultText):
    # ToDo: this function may take some further fine-tuning to show the text well given any possible image size

    imageHeight, imageWidth, sceneNumChannels = openCVImage.shape

    # choose a font
    fontFace = cv2.FONT_HERSHEY_TRIPLEX

    # chose the font size and thickness as a fraction of the image size
    fontScale = 1.0
    fontThickness = 2

    # make sure font thickness is an integer, if not, the OpenCV functions that use this may crash
    fontThickness = int(fontThickness)

    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)

    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize

    # calculate the lower left origin of the text area based on the text area center, width, and height
    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight

    # write the text on the image
    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, SCALAR_BLUE, fontThickness)
# end function

#######################################################################################################################

def Create_Dir(path):
    try:  
        os.makedirs(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s" % path)
        
#######################################################################################################################
def createConfusionMatrix(predictionsList, groundTruthtxt, C, run, what):
    groundTruthtxt = groundTruthtxt #'testSetGroundTruth24-29.txt'
    GroundTruthList = []
    predictionsList.sort()
    PredictedList = predictionsList
    
    with open(groundTruthtxt, 'r') as f:
        lines = f.readlines()
        lines.sort()
        GroundTruthList = lines
    # print(len(GroundTruthList))
    # print(len(PredictedList))
    y_true = []
    y_pred = []
    # print(len(GroundTruthList))
    # print(len(PredictedList))
    for i in range(len(GroundTruthList)):
        # print(GroundTruthList[i])
        gt = GroundTruthList[i].split(',')[-1]
        gt = gt.split('\n')[:-1]
        #print(PredictedList[i])
        pred = PredictedList[i].split(',')[-1]
        # print(gt)
        # print(pred)
        #pred = pred.split('\n')[:-1]
        y_true.append(C[gt[-1].lower()])
        #y_pred.append(C[pred[-1].lower()])
        if 'turnout' in pred.lower():
            y_pred.append(C['switch_closure_panel'])
        else:
            y_pred.append(C[pred.lower()])
        
    
    print(len(y_true), len(y_pred))
    
    cnf_matrix = confusion_matrix(y_true, y_pred)

    import warnings
    warnings.filterwarnings('ignore')
    # Plot normalized confusion matrix
    plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=C.keys(), normalize=True,
    #                      title=savename)
    plot_confusion_matrix_new(cnf_matrix, classes=C.keys(), normalize=True,
                          run=run,what=what)
                          
    #tmp = FileLocation1 + '.png'
    #plt.savefig(tmp, dpi=my_dpi)   # save the figure to file
    #plt.show()

    print("==== F1-Score ====")
    print(f1_score(y_true, y_pred, average='micro'))
    print(f1_score(y_true, y_pred, average='macro'))
    print(f1_score(y_true, y_pred, average='weighted'))

    # print("==== F1-Score ====")
    # print(f1_score(y_true, y_pred, average='micro'))
    # print(f1_score(y_true, y_pred, average='macro'))
    # print(f1_score(y_true, y_pred, average='weighted'))

    # print("==== F1-Score ====")
    # print(f1_score(y_true, y_pred, average='micro'))
    # print(f1_score(y_true, y_pred, average='macro'))
    # print(f1_score(y_true, y_pred, average='weighted'))


#######################################################################################################################
if __name__ == "__main__":
    tmp = main('', '', '', '', '', '', '')
