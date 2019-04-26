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


IMAGE_H, IMAGE_W = 608, 608
classes = utils.read_coco_names('./data/bane.names')
num_classes = len(classes)
image_path = "./bane_dataset/images/test.jpg"  # 181,
img = Image.open(image_path)
#make it balck and white
img=img.convert('L') 
img_resized = np.array(img.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
img_resized = img_resized / 255.
cpu_nms_graph = tf.Graph()

input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, "./checkpoint/yolov3bane_cpu_nms.pb",
                                           ["Placeholder:0", "concat_9:0", "mul_6:0"])

import cv2
import numpy as np

with tf.Session(graph=cpu_nms_graph) as sess:
    boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(np.expand_dims(img_resized, axis=0),axis=3)})
    print('boxes')
    print(boxes)
    #boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.05, iou_thresh=0.5)
    #boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
    #np.expand_dims(np.expand_dims(img_resized, axis=0),axis=3)
    boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)
    print('boxes')
    print(boxes)
    img = np.array(img_resized) 
# Convert RGB to BGR 
    #open_cv_image = open_cv_image[:, :, ::-1].copy()
    #cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
    #print('boxes')
    #print(boxes)
    #ret box
    cv2.rectangle(img,( int(boxes[0][0]),int(boxes[0][1])),( int(boxes[0][2]), int(boxes[0][3]) ),(0,255,0),3)
    
    cv2.rectangle(img,( int(boxes[1][0]),int(boxes[1][1])),( int(boxes[1][2]), int(boxes[1][3]) ),(0,255,0),3)
    
    factor = 1.8
        
    img = cv2.resize(img, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST) 
    
    cv2.imshow('img',img)
    cv2.waitKey(0)
    #image = draw_boxes_grayscale(img, boxes, scores, labels, classes, [IMAGE_H, IMAGE_W], show=True)
