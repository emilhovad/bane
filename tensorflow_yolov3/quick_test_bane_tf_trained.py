#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : fuck.py
#   Author      : YunYang1994
#   Created date: 2019-01-23 10:21:50
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
import cv2 as cv
import cv2
#import cv


IMAGE_H, IMAGE_W = 608, 608#416, 416
classes = utils.read_coco_names('./data/bane.names')
num_classes = len(classes)
image_path = "./straittrainingset/BTR_014034-H_km_36053_skinne-V_UIC211_FG2.jpg"  # 181,#BTR_012007-2_km_7451_skinne-V_UIC421_FG2.jpg
img = Image.open(image_path)
#show in normal format
factor = np.array(img).shape[0]/np.array(img).shape[1]
#img = Image.new("RGB", img.size) 
#make it balck and white
#img=img.convert('L') 
img_resized = np.array(img.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
img_resized = img_resized / 255.
cpu_nms_graph = tf.Graph()

#608
input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, "./checkpoint/yolov3_cpu_nms.pb",
                                           ["Placeholder:0", "concat_9:0", "mul_6:0"])

#416
#input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, "./checkpoint/yolov3_tf_416_bane_cpu_nms.pb",
#                                           ["Placeholder:0", "concat_9:0", "mul_6:0"])

import cv2
import numpy as np

with tf.Session(graph=cpu_nms_graph) as sess:
    boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
    #boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(np.expand_dims(img_resized, axis=0),axis=3)})
    #print('boxes')
    #print(boxes)
    #boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.05, iou_thresh=0.5)
    #boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
    #np.expand_dims(np.expand_dims(img_resized, axis=0),axis=3)
    boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.6, iou_thresh=0.4)
    print('boxes')
    print(boxes)
    
    print('labels')
    print(labels)
    
    print('scores')
    print(scores)
    
    img = np.array(img_resized) 
# Convert RGB to BGR 
    #open_cv_image = open_cv_image[:, :, ::-1].copy()
    #cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
    #print('boxes')
    #print(boxes)
    #ret box
    if labels is not None:
        for i in range(len(labels)):
            if labels[i]==0:
                cv2.rectangle(img,( int(boxes[i][0]),int(boxes[i][1])),( int(boxes[i][2]), int(boxes[i][3]) ),(255,0,0),1)
            if labels[i]==1:
                cv2.rectangle(img,( int(boxes[i][0]),int(boxes[i][1])),( int(boxes[i][2]), int(boxes[i][3]) ),(0,255,0),1)
            if labels[i]==2:
                cv2.rectangle(img,( int(boxes[i][0]),int(boxes[i][1])),( int(boxes[i][2]), int(boxes[i][3]) ),(0,0,255),1)
        
    #cv2.rectangle(img,( int(boxes[1][0]),int(boxes[1][1])),( int(boxes[1][2]), int(boxes[1][3]) ),(0,255,0),1)
    
    #factor = 1.8
        
    img = cv2.resize(img, (0,0), fx=1, fy=factor, interpolation=cv2.INTER_NEAREST) 
    
    cv2.imshow('img',img)
    cv2.waitKey(0)
    #image = draw_boxes_grayscale(img, boxes, scores, labels, classes, [IMAGE_H, IMAGE_W], show=True)
