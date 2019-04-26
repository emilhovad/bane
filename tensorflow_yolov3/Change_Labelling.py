#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 00:32:05 2019

@author: emil
"""
import os
import glob
import numpy as np
import cv2
from PIL import Image, ImageStat
#change labelling from tensorflow yolo to normal yolo

#a)
#make individual files with the corresponding image file name
current_path ='/home/emil/bane/tensorflow_yolov3/'
#new labelling for this frame work
pathlabel='/home/emil/bane/tensorflow_yolov3/tf_yolo_label/'

os.chdir(current_path)
path = '/straittrainingset/'

#create file
f_tf=open(pathlabel+"labelThomas.txt","w+")#"tennislabel.txt"
#f_tf.write(path+filename+" ")# +'\n')
f_tf.close()

#b)
#put all the image paths in on files with the content
os.chdir(current_path+path)
for filename in glob.glob("*.txt"):#os.listdir(current_path+path):
    #print(filename)
    yolo_file= open(filename,'r')
    img_path=current_path.rstrip("\/")+path+filename.rstrip("txt")+"jpg"
    img = cv2.imread(img_path)
    im= Image.open(img_path)
    rgbim  =  Image.new("RGB", im.size)    
    rgbim.paste(im)
    rgbim.save(current_path+'straittrainingset/'+filename.rstrip("txt")+"jpg")
    #stat=ImageStat.Stat(im)
    #stat.sum
    #    factor = 0.4
    #small = cv2.resize(im, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST) 
    
    
    if  os.path.getsize(filename)>0:
    
        f_tf= open(pathlabel+"labelThomas.txt",'a')      
        f_tf.write('.'+path+filename.rstrip("txt")+"jpg")#img_path)
        #f_tf.close()
    
        for line in yolo_file:#range(len(labels)):
            #lines.append([line])
            f_tf= open(pathlabel+"labelThomas.txt",'a')
            tf_line= (line.rstrip("\n")).split()
            
            if len(tf_line)>0:
                xl =  float(tf_line[1])*img.shape[1]-float(tf_line[3])*img.shape[1]/2 
                yt =  float(tf_line[2])*img.shape[0]-float(tf_line[4])*img.shape[0]/2 
                xr=   float(tf_line[1])*img.shape[1]+float(tf_line[3])*img.shape[1]/2 
                yb =  float(tf_line[2])*img.shape[0]+float(tf_line[4])*img.shape[0]/2 
        
                xl = ' '+str(xl)+' '
                yt = str(yt)+' '
                xr = str(xr)+' '
                yb = str(yb)+' '
                
                
                class_object= str(int(tf_line[0])) 
                
                f_tf.write(xl+yt+xr+yb+class_object)
        
        #f_tf.write(current_path+filename+" "+lines[:][:])
        f_tf.write('\n')      
        f_tf.close()
        
#        print(line)
#        
#        
#        if len(line)>0:
#            #person (look out for tennis)
#       
#            #tensorflow label
#            f_tf= open(pathlabel+"labelThomas.txt",'a')
#            f_tf.write(line+' ')
#            f_tf.close()

#and back 