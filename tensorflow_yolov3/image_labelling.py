#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
#================================================================

import colorsys
from PIL import ImageFont, ImageDraw

def draw_boxes_grayscale(image, boxes, scores, labels, classes, detection_size,
               font='./data/font/FiraMono-Medium.otf', show=True):
    """
    :param boxes, shape of  [num, 4]
    :param scores, shape of [num, ]
    :param labels, shape of [num, ]
    :param image,
    :param classes, the return list from the function `read_coco_names`
    """
    if boxes is None: return image
    draw = ImageDraw.Draw(image)
    # draw settings
    font = ImageFont.truetype(font = font, size = np.floor(2e-2 * image.size[1]).astype('int32'))
    hsv_tuples = [( x / len(classes), 0.9, 1.0) for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    for i in range(len(labels)): # for each bounding box, do:
        bbox, score, label = boxes[i], scores[i], classes[labels[i]]
        bbox_text = "%s %.2f" %(label, score)
        text_size = draw.textsize(bbox_text, font)
        # convert_to_original_size
        detection_size, original_size = np.array(detection_size), np.array(image.size)
        ratio = original_size / detection_size
        bbox = list((bbox.reshape(2,2) * ratio).reshape(-1))

        draw.rectangle(bbox, outline=colors[labels[i]], width=3)
        text_origin = bbox[:2]-np.array([0, text_size[1]])
        draw.rectangle([tuple(text_origin), tuple(text_origin+text_size)], fill=colors[labels[i]])
        # # draw bbox
        draw.text(tuple(text_origin), bbox_text, fill=(0,0,0), font=font)

    image.show() if show else None
    return image

import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils
import cv2
import os
import re


IMAGE_H, IMAGE_W = 608, 608
classes = utils.read_coco_names('./data/bane.names')
num_classes = len(classes)


cpu_nms_graph = tf.Graph()

input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, "./checkpoint/yolov3bane_cpu_nms.pb",
                                           ["Placeholder:0", "concat_9:0", "mul_6:0"])




#frame.shape
path = "./bane_dataset/images/"  # 181,
#img = Image.open(image_path)

pathlabel='/home/emil/tensorflow-yolov3-bane/tf_yolo_label/'
pathyololabel='/home/emil/tensorflow-yolov3-bane/yolo_label/'
framestart = 0
count_img=0
 


writeLine=True
with tf.Session(graph=cpu_nms_graph) as sess:

        #iterate over images
    for filename in os.listdir(path):    
        #print(filename)
        
        img = Image.open(path+filename)
        #make it balck and white
        img=img.convert('L') 
        img_resized = np.array(img.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
        img_resized = img_resized / 255.
        

        
        #write file for yolo labelling (Not tf yolo labelling)
        file = open(pathyololabel+re.sub('\.jpg$', '', filename)+ ".txt", "w") 
        

        boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(np.expand_dims(img_resized, axis=0),axis=3)})
    
        boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)
        

        if count_img==0 and writeLine:
            #create txt file for tf yolo label
            f= open(pathlabel+"label.txt","w+")#"tennislabel.txt"
            f.write(path+filename+" ")# +'\n')
            f.close()
            writeLine=False
        
        
        if count_img>0:# and writeLine:
            f= open(pathlabel+"label.txt",'a')#"tennislabel.txt"
            f.write('\n')
            f.write(path+filename+" ")# +'\n')
            f.close()
            writeLine=False
        
        

  
        #can write if no predictions exists
        if labels is not None:
            
            for i in range(len(labels)):
                
                #person (look out for tennis)
                if labels[i]==0:        
                    #tensorflow label
                    f= open(pathlabel+"label.txt",'a')
                    f.write(str( int(boxes[i][0]/1))+' '+str( int( boxes[i][1]/1) ) +' ' + str( int( boxes[i][2]/1 ) )+' '+str( int( boxes[i][3]/1) ) +' ' + str(0)+' ')
                    f.close()
                    #f.write(str(x)+' '+str(y) +' ' + str(2)+' ')
                    
                    ################################
                    #yolo txt file label
                    #<object-class> <x middle> <y middle> <width> <height>
                    #str( (boxes[i][0]/832))+' '+str( ( boxes[i][1]/832) ) +' ' + str( ( boxes[i][2]/832 ) )+' '+str( ( boxes[i][3]/832) )
                    xmiddle = str( ((boxes[i][0]+boxes[i][2])/IMAGE_W)/2 )
                    ymiddle = str( ((boxes[i][1]+boxes[i][3])/IMAGE_H)/2 )
                    xwidth  = str( ((boxes[i][2]-boxes[i][0])/IMAGE_W) )
                    yheight = str( ((boxes[i][3]-boxes[i][1])/IMAGE_H) )
                    file.write(str(0)+" "+xmiddle+' '+ymiddle+' ' +xwidth+' '+yheight )
                    #file.write(str(0)+" "+str( (boxes[i][0]/832))+' '+str( ( boxes[i][1]/832) ) +' ' + str( ( boxes[i][2]/832 ) )+' '+str( ( boxes[i][3]/832) ) )
                    file.write("\n")
                    
                #ball (look out for tennis)
                if labels[i]==1:        
                    f= open(pathlabel+"label.txt",'a')
                    f.write(str( int(boxes[i][0]/1))+' '+str( int( boxes[i][1]/1) ) +' ' + str( int( boxes[i][2]/1 ) )+' '+str( int( boxes[i][3]/1) )+' ' + str(0)+' ')
                    f.close()
                    ####################
                    #yolo txt file label
                    xmiddle = str( ((boxes[i][0]+boxes[i][2])/IMAGE_W)/2 )
                    ymiddle = str( ((boxes[i][1]+boxes[i][3])/IMAGE_H)/2 )
                    xwidth  = str( ((boxes[i][2]-boxes[i][0])/IMAGE_W) )
                    yheight = str( ((boxes[i][3]-boxes[i][1])/IMAGE_H) )
                    file.write(str(0)+" "+xmiddle+' '+ymiddle+' ' +xwidth+' '+yheight )
                    #file.write(str(1)+" "+str( (boxes[i][0]/832))+' '+str( ( boxes[i][1]/832) ) +' ' + str( ( boxes[i][2]/832) )+' '+str( ( boxes[i][3]/832) ) )
                    file.write("\n")
                    
            
                    
            #close yolo label txt file
            file.close()
            
        count_img += 1            
    
            
        if False:
            img = np.array(img_resized) 

            cv2.rectangle(img,( int(boxes[0][0]),int(boxes[0][1])),( int(boxes[0][2]), int(boxes[0][3]) ),(0,255,0),3)
            
            cv2.rectangle(img,( int(boxes[1][0]),int(boxes[1][1])),( int(boxes[1][2]), int(boxes[1][3]) ),(0,255,0),3)
            
            factor = 1.8
                
            img = cv2.resize(img, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST) 
            
            cv2.imshow('img',img)
            cv2.waitKey(0)
        
        
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

