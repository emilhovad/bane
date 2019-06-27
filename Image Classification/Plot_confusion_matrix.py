import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    # fmt = '.2f' if normalize else 'd'
    # #thresh = cm.max() / 2.
    # thresh = np.nanmax(cm)/2
    # #thresh = max(map(max, cm))/2.0
    # print(thresh)
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i, format(cm[i, j], fmt),
                 # horizontalalignment="center",
                 # color="white" if cm[i, j] > thresh else "k")

    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.tight_layout()


# Compute confusion matrix
#cnf_matrix = confusion_matrix(y_test, y_pred)
#np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names,
#                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix')

#plt.show()

#new plot

def plot_confusion_matrix_new(cm, classes,
                          normalize=False,
                          run='name', what='train',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    #run = "Run4C-1"
    #what = "train"
    
    #title = run + "; " + what + " set"
    title = what + " set"
    #classes = ["NoClass","UIC211","UIC227","UIC411","UIC421"]

    font = {'family':'serif', 'weight': 'normal', 'size': 16}
    font_ticks = {'family':'serif', 'weight': 'normal', 'size': 16, 'color': '#2F4F4F'}
    font_title = {'family':'serif', 'weight': 'bold', 'size': 20}

    f = plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title(title, **font_title)

    #plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, **font_ticks)
    plt.yticks(tick_marks, classes, rotation=45, **font_ticks)

    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black", **font)

    plt.ylabel('True label', **font)
    plt.xlabel('Predicted label', **font)

    f.savefig('ConfusionMatrixs/26_06_2019/' + run + "_" + what + ".pdf", bbox_inches='tight')
    f.savefig('ConfusionMatrixs/26_06_2019/' + run + "_" + what + ".png", bbox_inches='tight',dpi=250)

    
def calculateRecallPrecisionAndF1Score(y_true, y_pred):
    result = [['Function','micro','macro','weighted']]
    tmp = ['F1_score']
    tmp.append(f1_score(y_true, y_pred, average='micro'))
    tmp.append(f1_score(y_true, y_pred, average='macro'))
    tmp.append(f1_score(y_true, y_pred, average='weighted'))
    result.append(tmp)
    tmp = ['recall_score']
    tmp.append(recall_score(y_true, y_pred, average='micro'))
    tmp.append(recall_score(y_true, y_pred, average='macro'))
    tmp.append(recall_score(y_true, y_pred, average='weighted'))
    result.append(tmp)
    tmp = ['precision_score']
    tmp.append(precision_score(y_true, y_pred, average='micro'))
    tmp.append(precision_score(y_true, y_pred, average='macro'))
    tmp.append(precision_score(y_true, y_pred, average='weighted'))
    result.append(tmp)
    # print("==== F1-Score ====")
    # print(f1_score(y_true, y_pred, average='micro'))
    # print(f1_score(y_true, y_pred, average='macro'))
    # print(f1_score(y_true, y_pred, average='weighted'))
            
    # print("==== recall_score ====")
    # print(recall_score(y_true, y_pred, average='micro'))
    # print(recall_score(y_true, y_pred, average='macro'))
    # print(recall_score(y_true, y_pred, average='weighted'))

    # print("==== precision_score ====")
    # print(precision_score(y_true, y_pred, average='micro'))
    # print(precision_score(y_true, y_pred, average='macro'))
    # print(precision_score(y_true, y_pred, average='weighted'))
    
    return result
    