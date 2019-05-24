
# coding: utf-8

# In[1]:


from PIL import Image
import glob
import random
import os.path
import shutil
import os


# In[7]:


image_list_valid = []
image_list_train = []
dst_dir_test = "ToAddUIC211WithTxt"
for filename in glob.glob('ToTest/*.jpg'): #assuming gif
    if os.path.isfile(filename[:-3] + "txt"):
        with open(filename[:-3] + "txt", 'r') as content:
            annotations = content.read()
            #print(annotations)
            if len(annotations) > 0:
                #filename = os.path.split(filename)[-1]
                #print(filename)
                shutil.copy(filename, dst_dir_test)
                filename = filename[:-3] + "txt"
                #print(filename)
                shutil.copy(filename, dst_dir_test)


# In[11]:


