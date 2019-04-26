# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:42:40 2019

@author: emilh
"""

import cv2
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import colors
import itertools

from PIL import Image
import glob
from sklearn.cluster import KMeans
    

image_list_0 = []
image_list_1 = []

summary_list_0 = []
summary_list_1 = []


railwidth = 50#64
convfilter = np.ones((railwidth))

#Program a sleeper cutter.
def ImageSegmentation(im,railwidth):
    
    #Finding the stones versus rails based on low and high difference in 
    #column pixels values
    SumDifferenceCol = np.sum(im[1::1,:][0:-1,:]-im[2::1,:],axis=0)
    
    
    padding = np.ones(( np.int( railwidth/2)))*10e12
    padded_vector = []
    padded_vector.append(padding[:].tolist())  
    padded_vector.append(SumDifferenceCol[:].tolist())  
    padded_vector.append(padding[:-1].tolist())  
    padded_vector = list(itertools.chain(*padded_vector))
    padded_vector = np.asarray(padded_vector)
    
    #Finding the best fit, the rails has a lower column standard deviation 
    #as compared to the stones 
    #make a convolution with the rail width
    convfilter = np.ones((railwidth))
    
    #Finding the lowest value for the two rails based on a convolution
    convResult = np.convolve( padded_vector  , convfilter[:], 'valid')
    #convResult.shape
    
    np.argpartition(convResult,6)
    
    sortedlist = np.argsort(convResult)[:(railwidth*2)]
        
    # Initializing KMeans
    kmeans = KMeans(n_clusters=2)
    # Fitting with inputs
    kmeans = kmeans.fit(sortedlist.reshape(-1, 1))
    # Predicting the clusters
    labels = kmeans.predict(sortedlist.reshape(-1, 1))
    # Getting the cluster centers
    C = kmeans.cluster_centers_
    
    rail1start=np.int(C[0]-railwidth/2)
    rail1end  = np.int(C[0]+railwidth/2)

    rail2start=np.int(C[1]-railwidth/2)
    rail2end  = np.int(C[1]+railwidth/2)
    
    #imgCut1=Image.fromarray(im[:,rail1start:rail1end])
    imgCut1=im[:,rail1start:rail1end]
    imgCut2=im[:,rail2start:rail2end]
    
    return imgCut1, imgCut2

#Program a sleeper cutter.
    
#lack normalization 
def CustomFilteringmethod(imgCut1):
    
    col_mean1 = np.mean(imgCut1,axis=0)
    col_sd1 = np.std(imgCut1,axis=0)
    
    y = np.zeros(( imgCut1.shape[0] ,  imgCut1.shape[1] ))
    y_fault  = np.abs(np.abs(imgCut1-col_mean1)-col_sd1)
    larger_than= y_fault >  np.zeros(( imgCut1.shape[0] ,  imgCut1.shape[1] ))
    y[larger_than] = y_fault[larger_than]
    y = np.asarray(y, dtype=np.uint8) #y.astype(uint8)#(int)# dtype=uint8
    return y


def RailResults(yL,yR):
    max_values = ( np.max(yL), np.max(yR))
    std_values = ( np.std(yL), np.std(yR))
    
    #use the maximum and the std for the same rail
    indice_std = max_values.index(max(max_values))  
    std_value = std_values[indice_std]     
    max_value=max( max_values)
    return max_value, std_value


countplot=0
for filename in glob.glob('H:/BaneDk/WorkFiles/WorkFiles/Images/Examplesof0/*.jpg'): #assuming gif
    im = cv2.imread(filename,0)
    image_list_0.append(im)
    
    # Segmentation
    imgCutL, imgCutR = ImageSegmentation(im,railwidth)


  
    #plot rail left and rail right    
    if False:
        #Rail 1
        type(imgCut1)
        imgCut1.shape
        imgCut1
        #y
        imgCut1Plot=Image.fromarray(imgCut1)
        type(imgCut1Plot)
        imgCut1Plot
        
        #rail 2
        type(imgCut2)
        imgCut2.shape
        imgCut2
        #y
        imgCut2Plot=Image.fromarray(imgCut2)
        type(imgCut2Plot)
        imgCut2Plot
        
    
    # normalize image before filtering method ?
    if False:
        blurL = cv2.blur(imgCutL,(5,5))
        blurR = cv2.blur(imgCutR,(5,5))
        
               
        kernel = np.ones((5,5),np.uint8)
        
        tophatL = cv2.morphologyEx(imgCutL, cv2.MORPH_TOPHAT, kernel)
        tophatR = cv2.morphologyEx(imgCutR, cv2.MORPH_TOPHAT, kernel)
        
        tophatRPlot=Image.fromarray(255-tophatR)
        type(tophatRPlot)
        tophatRPlot
        
        blackhatL = cv2.morphologyEx(imgCutL, cv2.MORPH_BLACKHAT, kernel)
        blackhatR = cv2.morphologyEx(imgCutR, cv2.MORPH_BLACKHAT, kernel)
        
        blackhatRPlot=Image.fromarray(255-blackhatR)
        type(blackhatRPlot)
        blackhatRPlot
        
        imgCutLsobelx = cv2.Sobel(imgCutL,cv2.CV_64F,1,0,ksize=9)
        imgCutLsobely = cv2.Sobel(imgCutL,cv2.CV_64F,0,1,ksize=9)
        
        cv2.imshow('sobel',imgCutLsobely)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  
        
        imgCutLsobelxRPlot=Image.fromarray(imgCutLsobelx)
        imgCutLsobelxRPlot
        
        imgplot = plt.imshow(imgCutLsobelx)
        

        
        #plt.hist(imgCutLsobelx.ravel(),256,[0,256]); plt.show()
        
        #cv2.imshow('img',255-blackhatR)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()            
        
    # Filtering method      
    yL = CustomFilteringmethod(imgCutL)
    yR = CustomFilteringmethod(imgCutR)
    
    
    plt.subplot(1, 2, 1)#+countplot)
    plt.imshow(yL,cmap='Greys')
    plt.title('left rail, group 0')
    plt.subplot(1, 2, 2)#+countplot)
    plt.title('right rail, group 0')
    plt.imshow(yR, cmap='Greys')
    
    plt.show()
    
    countplot+=1
    #get the results from the correct rail
    max_value, std_value = RailResults(yL,yR)
    
    #append results
    summary_list_0.append([max_value, std_value])
    
    #plot code
    if False:
        yLPlot=Image.fromarray(255-yL)
        yLPlot
    
        yRPlot=Image.fromarray(255-yR)
        yRPlot
    
countplot=0 
for filename in glob.glob('H:/BaneDk/WorkFiles/WorkFiles/Images/Examplesof1/*.jpg'): #assuming gif
    im = cv2.imread(filename,0)
    #im=Image.open(filename)
    image_list_1.append(im)
    
     # Segmentation
    imgCutL, imgCutR = ImageSegmentation(im,railwidth)
    
    # Filtering method      
    yL = CustomFilteringmethod(imgCutL)
    yR = CustomFilteringmethod(imgCutR)
    
    plt.subplot(1, 2, 1)#+countplot)
    plt.imshow(yL,cmap='Greys')
    plt.title('left rail, group 1')
    plt.subplot(1, 2, 2)#+countplot)
    plt.title('right rail, group 1')
    plt.imshow(yR, cmap='Greys')
    
    plt.show()
    
    countplot+=1
    
    #get the results from the correct rail
    max_value, std_value = RailResults(yL,yR)
    
    #append results
    summary_list_1.append([max_value, std_value])



#Show results 
summary_0 = np.asarray(summary_list_0)    
summary_1 = np.asarray(summary_list_1)    

max_0 = max(summary_0[:,0])  
max_1 = max(summary_1[:,0])    


mean_0 = np.mean(summary_0[:,0])  
mean_1 = np.mean(summary_1[:,0])    

std_0 = np.std(summary_0[:,0])  
std_1 = np.std(summary_1[:,0])    

print("max 0:", max_0, ", max 1:", max_1, ", mean 0:", mean_0, ", mean 1:", mean_1, ", std 0:", std_0, ", std 1:", std_1)
    
#Normalization group 0
summary_0[:,0] = (summary_0[:,0]-max_0)/(max_1-max_0)


#Normalization group 1
summary_1[:,0] = (summary_1[:,0]-max_0)/(max_1-max_0)    

    


    #img = cv2.imread('H:/BaneDk/WorkFiles/WorkFiles/Images/Examplesof0/Pic2c0136bf52ce4e1eae2ea343a32fc4fb.jpg',0)
     #plt.imshow(yLPlot)    
    #cv2.imshow("left rail",yL)    
    #yRPlot=Image.fromarray(255-yR)
    #yRPlot
    #cv2.waitKey(0)
    #cv2.destroyWindow('i')    
    #imgCut2Plot
    #y = np.max( np.abs(np.abs(imgCut1-col_mean1)-col_sd1),  np.zeros(( imgCut1.shape[0] ,  imgCut1.shape[1] )) )
    #y.shape
    #imgCut1-col_mean1    
#    imgCut2=im[:,rail2start:rail2end]
#    type(imgCut2)
#    imgCut2.shape
#    imgCut2=Image.fromarray(imgCut2)      
        

    
    #type(imgCut2)

    #col_mean2 = np.mean(imgCut2,axis=0)
    #imgCut2-col_mean2    
    #imgCut2=Image.fromarray(imgCut2)
    #imgCut2
        
    #im=Image.open(filename)
    
    #np.int(C[1])

    
    #firstrail  = sortedlist[np.argsort(sortedlist)][0:railwidth]
    
    #get the meadian
    
    #secondrail = sortedlist[np.argsort(sortedlist)][(railwidth+1):(railwidth*2)]
    
    #get the median
    
    #np.where(np.argmin(convResult))   
    #padded_vector =   , SumDifferenceCol[:].flatten()[:], padding[:].flatten()[:]] )
    #np.concatenate(padding[:].flatten(), SumDifferenceCol[:].flatten(), padding[:].flatten())
    #if im.shape[0] % 2 == 1:    
    #SumDifferenceCol = np.sum(im[1::2,:]-im[2::2,:],axis=0)    
    #find the width of the rail in pixels
    #SumDifferenceCol.shape
    #np.Inf((railwidth))
    #left = np.zeros((1,20))
    #right = left = np.zeros((1,20))
    #SumDifferenceCol[1::railwidth]
    #im=Image.open(filename)
    #image_list_0.append(im)
    #img=Image.open(filename)
    #imgCut=Image.fromarray(im[:,405:470])
    #imgCut
    #cv2.imshow('image',im)
    #plt.show(im[1::2,:].shape)
#    im.shape
#    
#    im[1::2,0].shape
#    im[2::2,0].shape
    
    
    

#    cv2.imshow('frame', im)
#    plt.plot(SumDifference)
#    plt.ylabel('some numbers')
#    plt.show()
#    
#    
#    plt.plot(im)
#    plt.ylabel('some numbers')
#    plt.show()
    #im[1::2,:][1:-1,:].shape
    #im[1::2,:]-im[2::2,:].shape
    
    #im[,]-
 
#cv2.imshow('image',im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()    
 
   
 
#w=600
#h=1300
#fig=plt.figure()
#for i in range(1,20):
#    img = image_list_1[i]#np.random.randint(10, size=(h,w))
#    fig.add_subplot(i,2,1)
#    plt.imshow(img)
#plt.show()
#plt.show( image_list_1[0])

#image_list_0[0].shape

#col_mean = np.mean(image_list_0[0],axis=0)
#col_mean.shape
    
    #    if False:
#        SumDifferenceCol = np.sum(im[1::1,:][0:-1,:]-im[2::1,:],axis=0)
#        
#    
#        padding = np.ones(( np.int( railwidth/2)))*10e12
#        padded_vector = []
#        padded_vector.append(padding[:].tolist())  
#        padded_vector.append(SumDifferenceCol[:].tolist())  
#        padded_vector.append(padding[:-1].tolist())  
#        padded_vector = list(itertools.chain(*padded_vector))
#        padded_vector = np.asarray(padded_vector)
#        
#        convResult = np.convolve( padded_vector  , convfilter[:], 'valid')
#       
#        convResult.shape
#        
#        np.argpartition(convResult,6)
#        
#        sortedlist = np.argsort(convResult)[:(railwidth*2)]
#            
#        # Initializing KMeans
#        kmeans = KMeans(n_clusters=2)
#        # Fitting with inputs
#        kmeans = kmeans.fit(sortedlist.reshape(-1, 1))
#        # Predicting the clusters
#        labels = kmeans.predict(sortedlist.reshape(-1, 1))
#        # Getting the cluster centers
#        C = kmeans.cluster_centers_
#        
#        rail1start=np.int(C[0]-railwidth/2)
#        rail1end  = np.int(C[0]+railwidth/2)
#    
#        rail2start=np.int(C[1]-railwidth/2)
#        rail2end  = np.int(C[1]+railwidth/2)
#        
#        #imgCut1=Image.fromarray(im[:,rail1start:rail1end])
#        imgCut1=im[:,rail1start:rail1end]
#        type(imgCut1)
#        imgCut1.shape  

#    col_mean1 = np.mean(imgCut1,axis=0)
#    col_sd1 = np.std(imgCut1,axis=0)
#    
#    #type(imgCut1)
#    
#    imgCut1.shape
#    col_mean1.shape
#    #(np.abs(np.abs(imgCut1-col_mean1)-col_sd1)).shape
#    y = np.zeros(( imgCut1.shape[0] ,  imgCut1.shape[1] ))
#    y_fault  = np.abs(np.abs(imgCut1-col_mean1)-col_sd1)
#    larger_than= y_fault >  np.zeros(( imgCut1.shape[0] ,  imgCut1.shape[1] ))
#    y[larger_than] = y_fault[larger_than]
#    y = y.astype(int)
    
    #    max_values = ( np.max(yL), np.max(yR))
#    std_values = ( np.std(yL), np.std(yR))
#    
#    #use the maximum and the std for the same rail
#    indice_std = values.index(max(values))  
#    std_value = std_values[indice_std]     
#    max_value=max( max_values)