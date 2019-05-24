from PIL import Image
import glob
import random
import os

currentPath = os.getcwd()
RunFolder = currentPath.split('\\')[-1]
#print(currentPath)

def createList(Dest):
    ToTest = []
    for filename in glob.glob(Dest + '/*.jpg'):
        tmp = filename.split('\\')
        tmp = tmp[0] + '/' + tmp[1]
        ToTest.append(tmp)
        
    with open('./' + Dest + '.txt', 'w', encoding='utf-8') as f:
        for item in ToTest:
            whatToSave = "data/BaneDk/" + str(RunFolder) + "/" + str(item) + "\n"
            f.write(whatToSave)

createList('TrainSet_MIC_Augmented_H_V_HV_3Classes')
createList('TrainSet_MIC_Augmented_H_V_HV_3Classes_Com')
createList('TrainSetAugmented_3Classes')
createList('TrainSetAugmented_3Classes_Com')
createList('ValidationSet_3Classes')
createList('ValidationSet_3Classes_Com')
createList('ValidationSet_MIC_3Classes')
createList('ValidationSet_MIC_3Classes_Com')
createList('TestSet_3Classes')
createList('TestSet_3Classes_Com')
createList('TestSet_MIC_3Classes')
createList('TestSet_MIC_3Classes_Com')