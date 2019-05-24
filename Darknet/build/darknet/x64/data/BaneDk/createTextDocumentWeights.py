from PIL import Image
import glob
import random
import os

currentPath = os.getcwd()
RunFolder = currentPath.split('\\')[-2]
#print(currentPath)

weightsList = []
for filename in glob.glob('*.weights'):
	newFile = filename.split("_")
	end = newFile[-1].split(".")[0]
	filename = "" + str(newFile[0]) + "_" + str(newFile[1]) + "_" + str(newFile[2]) + "_" + end
	weightsList.append(filename)

weightsList.sort(key = lambda x: int(x.split("_")[-1]))
	
with open('./allWeightsInFolder.txt', 'w') as f:
	for item in weightsList:
		whatToSave = "data/BaneDk/"+ RunFolder +"/backup/" + str(item) + ".weights\n"
		f.write(whatToSave)