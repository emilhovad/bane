from PIL import Image
import glob
import random
import os
import shutil
import codecs

ToTest = []
for filename in glob.glob('*.jpg'):
	ToTest.append(filename)
for filename in glob.glob('NoClass/*.jpg'):
    print(filename)
    tmp = filename.split('\\')
    tmp = tmp[0] + '/' + tmp[1]
    print(tmp)
    ToTest.append(tmp)
	#shutil.move(filename, "tmp/")
	#shutil.move(txtFilename, "tmp/")
		
	
with open('./ToTest.txt', 'w', encoding='utf-8') as f:
	for item in ToTest:
		whatToSave = "data/BaneDk/TestSet/test3Classes25_02_19/" + str(item) + "\n"
		f.write(whatToSave)