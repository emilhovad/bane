import os, re
import glob

LabelLocation = 'Labels/'#'train3Classes2.txt'
SaveName = 'Result.csv'

def is_not_blank(s):
    return bool(s and s.strip())  
def FindUIC227FromTxt(Label_Location):
    ImageResult = []
    train_imgs = []
    #Goes though each .txt file in Label_Location
    for filename in glob.glob(Label_Location + '*.txt'):
        LeftCount = 0
        RigthCount = 0
        Name = filename.split('\\')[-1]
        #gets the name of the file
        Name = Name.split('/')[-1]
        if os.path.isfile(filename):
            #opens the file
            with open(filename, 'r') as content:
                #split the file in to lines
                annotations = content.read().split('\n')#
                #check if the file is empty
                if len(annotations) == 0:
                    #if it is the name of the file plus that there are no UIC227 is added to the list
                    ImageResult.append([Name,"0","0"])
                else:
                    #goes though each line of the txt file
                    for annotation in annotations:
                        if is_not_blank(annotation):
                            #split the line 
                            pred = annotation.split(' ')
                            #check if the class is UIC211
                            if int(pred[0]) == 0:
                                #check if the class is at the left
                                if float(pred[1]) < 0.4:
                                    LeftCount += 1
                                #check if the class is at the left
                                elif float(pred[1]) > 0.6:
                                    RigthCount += 1
                    #both sides has a UIC227
                    if LeftCount >= 3 and RigthCount >= 3:
                        ImageResult.append([Name,"1","1"])
                    #left sides has a UIC227
                    elif LeftCount >= 3:
                        ImageResult.append([Name,"1","0"])
                    #right sides has a UIC227
                    elif RigthCount >= 3:
                        ImageResult.append([Name,"0","1"])
                    #no sides has a UIC227
                    else:
                        ImageResult.append([Name,"0","0"])
    return ImageResult

def SaveToFilename(resultArray, SaveName):
    #opens file to save to (Filename = SaveName)
    with open(SaveName, 'w') as f:
        #what first line should contain
        whatToSave = 'Name, UIC227 Left side, UIC227 right side'
        #goes though the result from function FindUIC227FromTxt
        for item in resultArray:
            #saves the part of the result
            whatToSave += item[0] + ', ' + item[1] + ', ' + item[2] + '\n'
        #saves it to the file
        f.write(whatToSave)    

#Runs the two function
result = FindUIC227FromTxt(LabelLocation)
SaveToFilename(result, SaveName)
