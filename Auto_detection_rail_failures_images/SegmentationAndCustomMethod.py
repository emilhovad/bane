# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:42:40 2019

@author: emilh
"""

import cv2
import cv2 as cv 
import numpy as np 
import matplotlib.mlab as mlab
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

#Program a sleeper cutter?

#rail image segmentation function
def ImageSegmentation(im,railwidth):
    
    #Finding the stones versus rails based on low and high difference in 
    #column pixels values
    SumDifferenceCol = np.sum(np.abs(im[1::1,:][0:-1,:]-im[2::1,:]),axis=0)
    #np.sum(im[1::1,:][0:-1,:]-im[2::1,:],axis=0)
    
    #check 15-200=?
    
    #padding?
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
    #np.argpartition(convResult,6)
    
    #list rail indices
    sortedlist = np.argsort(convResult)[:(railwidth*2)]
        
    # Initializing KMeans
    #
    startpts=np.array([[int(np.round(im.shape[1]/4))],[int(np.round(im.shape[1]*3/4))]])
    kmeans = KMeans(n_clusters=2, init=startpts)
    # Fitting with inputs
    kmeans = kmeans.fit(sortedlist.reshape(-1, 1))
    # Predicting the clusters
    #labels = kmeans.predict(sortedlist.reshape(-1, 1))
    # Getting the cluster centers
    C = kmeans.cluster_centers_
    
    rail1start=np.int(C[0]-railwidth/2)
    rail1end  = np.int(C[0]+railwidth/2)

    rail2start=np.int(C[1]-railwidth/2)
    rail2end  = np.int(C[1]+railwidth/2)
    
    #imgCut1=Image.fromarray(im[:,rail1start:rail1end])
    imgCut1=im[:,rail1start:rail1end]
    imgCut2=im[:,rail2start:rail2end]
    
    return imgCut1, imgCut2, rail1start, rail1end, rail2start, rail2end, convResult 

#Program a sleeper cutter.
    
#lack normalization ?
def CustomFilteringmethod(imgCut1):
    
    col_mean1 = np.mean(imgCut1,axis=0)
    col_sd1 = np.std(imgCut1,axis=0)
    
    y = np.zeros(( imgCut1.shape[0] ,  imgCut1.shape[1] ))
    y_fault  = np.abs(np.abs(imgCut1-col_mean1)-col_sd1)
    larger_than= y_fault >  np.zeros(( imgCut1.shape[0] ,  imgCut1.shape[1] ))
    y[larger_than] = y_fault[larger_than]
    y = np.asarray(y, dtype=np.uint8) #y.astype(uint8)#(int)# dtype=uint8
    return y

#def TopHatFilter(imgCut1):
#    kernel = np.ones((5,5),np.uint8)
#    y = cv2.morphologyEx(imgCut1, cv2.MORPH_TOPHAT, kernel)
#    return y
#
#def SabolFilter(imgCut1):
#    y = cv2.Sobel(imgCutL,cv2.CV_64F,1,0,ksize=9)
#    return y


#def build_filters():
#    """ returns a list of kernels in several orientations
#    """
#    filters = []
#    ksize = 31
#    for theta in np.arange(0, np.pi, np.pi / 32):
#        params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':15.0,
#                  'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
#        kern = cv2.getGaborKernel(**params)
#        kern /= 1.5*kern.sum()
#        filters.append((kern,params))
#    return filters
#
#def process(img, filters):
#    """ returns the img filtered by the filter list
#    """
#    accum = np.zeros_like(img)
#    for kern,params in filters:
#        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
#        np.maximum(accum, fimg, accum)
#    return accum


#filters = build_filters()
#p = process(imgCutL, filters)
#cv2.imshow('p',p)
#cv2.waitKey()
#cv2.destroyAllWindows
#cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
#    if False:
#        blurL = cv2.blur(imgCutL,(5,5))
#        blurR = cv2.blur(imgCutR,(5,5))
#        
#               
#        kernel = np.ones((5,5),np.uint8)
#        
#        tophatL = cv2.morphologyEx(imgCutL, cv2.MORPH_TOPHAT, kernel)
#        tophatR = cv2.morphologyEx(imgCutR, cv2.MORPH_TOPHAT, kernel)
#        
#        tophatRPlot=Image.fromarray(255-tophatR)
#        type(tophatRPlot)
#        tophatRPlot
#        
#        blackhatL = cv2.morphologyEx(imgCutL, cv2.MORPH_BLACKHAT, kernel)
#        blackhatR = cv2.morphologyEx(imgCutR, cv2.MORPH_BLACKHAT, kernel)
#        
#        blackhatRPlot=Image.fromarray(255-blackhatR)
#        type(blackhatRPlot)
#        blackhatRPlot
#        
#        imgCutLsobelx = cv2.Sobel(imgCutL,cv2.CV_64F,1,0,ksize=9)
#        imgCutLsobely = cv2.Sobel(imgCutL,cv2.CV_64F,0,1,ksize=9)
#        
#        
#        imgCutLsobelxRPlot=Image.fromarray(imgCutLsobelx)
#        imgCutLsobelxRPlot
#        
#        imgplot = plt.imshow(imgCutLsobelx)    
#

def RailResults(yL,yR):
    max_values = ( np.max(yL), np.max(yR))
    mean_values = ( np.mean(yL), np.mean(yR))
    std_values = ( np.std(yL), np.std(yR))
    j_std_values = ( np.max( np.std(yL,axis=0) ), np.max( np.std(yR, axis=0) ) )
    
    #use the maximum and the std for the same rail
    #indice_std = max_values.index(max(max_values)) 
    indice_std = j_std_values.index(max(j_std_values)) 
    std_value = std_values[indice_std]
    mean_value = mean_values[indice_std]     
    max_value=max( max_values)
    #line above is equal to = max_values[indice_std]
    return max(j_std_values), indice_std








from scipy.signal import convolve2d
#test, works
#filter_kernel = np.ones([3,3])/(9)
#Img = [[3,3,3,0,0,0],[3,3,3,0,0,0],[3,3,3,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
#test = convolve2d(Img, filter_kernel, mode='valid')
def stdFilter(imgCut1):
# STD FILTER, SMOOTHING, THRESHOLDING
    #kernel = (25,25)
    filter_kernel = np.ones([12,2])/(12*2)

    I_double = imgCut1.astype(np.double)#/256
    
    E = convolve2d(I_double**2, filter_kernel, mode='valid')-convolve2d(I_double, filter_kernel, mode='valid')    
    #E = np.sqrt(cv.boxFilter(I_double**2,-1,kernel)-cv.boxFilter(I_double,-1,kernel)**2)
    #mm=np.mean(E,axis=0)
    #cv2.imshow('img',E)
    #plt.imshow(E)
    #S = cv.GaussianBlur(E,(29,29),50)
    #T = S>0.025
    return E
    # VISUALIZATION
    #plt.imshow(imgCutL,cmap="gray")
#    fig = plt.figure(figsize=(12,5))
#    subfig = fig.add_subplot(1,3,1)
#    plt.imshow(E, cmap="gray")
#    subfig.set_title('Filtered')
#    subfig = fig.add_subplot(1,3,2)
#    plt.imshow(S, cmap="gray")
#    subfig.set_title('Smoothed')
#    subfig = fig.add_subplot(1,3,3)
#    plt.imshow(T, cmap="gray")
#    subfig.set_title('Thresholded')
#    fig.tight_layout()

#%%

method = "stdFilter"#'Custom'#'stdFilter'# "MaxstdColumn"

#loop over group 0
countplot=0
for filename in glob.glob('./Images/Examplesof0/*.jpg'): #assuming gif
    im = cv2.imread(filename,0)
    image_list_0.append(im)
    
    # Segmentation
    imgCutL, imgCutR, rail1start, rail1end, rail2start, rail2end, convResult = ImageSegmentation(im,railwidth)
    
    #get the results from the correct rail
    std_column_value, railside = RailResults(imgCutL,imgCutL)
   
    #plot rail left and rail right    
    if False:
        Image.fromarray(im)
        
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
        
    
    #investigate with anova?
    # normalize image before filtering method?.. 
 

        
        #plt.hist(imgCutLsobelx.ravel(),256,[0,256]); plt.show()
        
        #cv2.imshow('img',255-blackhatR)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows() 
    if method == "MaxstdColumn":
        #no difference but this is consistent with the code
        yL = std_column_value
        yR = std_column_value
           
    if method == "Custom":    
        # Filtering method      
        yL = CustomFilteringmethod(imgCutL)
        yR = CustomFilteringmethod(imgCutR)
    
    if method == "stdFilter":    
        # Filtering method      
        yL = stdFilter(imgCutL)
        yR = stdFilter(imgCutR)
    
    if False:       
        from statsmodels.graphics.gofplots import qqplot
        from scipy.stats import shapiro
        from scipy.stats import normaltest
        from scipy import stats
        
        for i in range(imgCutL.shape[1]):
            #plt.hist(imgCutL[:,i])
            #plt.show()
            #stat, p = shapiro(imgCutL[:,i])
            stat, p = normaltest(imgCutR[:,i])
            kolmosmirnoff=stats.kstest(imgCutR[:,i], 'norm')
            print(kolmosmirnoff)
            #print(p)
            qqplot(imgCutL[:,i], line='s')
            plt.show()

        
    
    
#    cv2.imshow('rail',yL)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    if not method=='MaxstdColumn':
        plt.subplot(1, 2, 1)#+countplot)
        plt.imshow(yL,cmap='Greys')
        plt.title('left rail, group 0')
        plt.subplot(1, 2, 2)#+countplot)
        plt.title('right rail, group 0')
        plt.imshow(yR, cmap='Greys')
        
        plt.show()
    
    countplot+=1

    
    #left rail
    if railside==0:
        max_value=np.max(yL)
        mean_value=np.mean(yL)
        std_value=np.std(yL)
        #add values to the summary list
        summary_list_0.append([max_value, mean_value, std_value])
    
    #right rail
    if railside==1:
        max_value=np.max(yR)
        mean_value=np.mean(yR)
        std_value=np.std(yR)
        #add values to the summary list
        summary_list_0.append([max_value, mean_value, std_value])
    
    
    
    #plot rail segmentation
    if  not method== 'stdFilter':
        im_rgb = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
        cv2.imwrite( "./opencvresults/group0/"+str(countplot)+'normal.png',im_rgb)
        
        im_rgb[:,:,:] = 0 
        
        #draw faults with red 
        im_rgb[0:im_rgb.shape[0],rail1start:rail1end,0] = yL
        im_rgb[0:im_rgb.shape[0],rail1start:rail1end,1] = yL
        im_rgb[0:im_rgb.shape[0],rail1start:rail1end,2] = yL
    
        im_rgb[0:im_rgb.shape[0],rail2start:rail2end,0] = yR
        im_rgb[0:im_rgb.shape[0],rail2start:rail2end,1] = yR
        im_rgb[0:im_rgb.shape[0],rail2start:rail2end,2] = yR
    
        im_rgb = cv2.bitwise_not(im_rgb)
    
        if railside==0: 
            cv2.line(im_rgb,(rail1start,0),(rail1start,1271),(255,0,0),3)
            cv2.line(im_rgb,(rail1end,0),(rail1end,1271),(255,0,0),3)
        
            cv2.line(im_rgb,(rail2start,0),(rail2start,1271),(255,0,0),2)
            cv2.line(im_rgb,(rail2end,0),(rail2end,1271),(255,0,0),2)
            
        if railside==1: 
            cv2.line(im_rgb,(rail1start,0),(rail1start,1271),(255,0,0),2)
            cv2.line(im_rgb,(rail1end,0),(rail1end,1271),(255,0,0),2)
        
            cv2.line(im_rgb,(rail2start,0),(rail2start,1271),(255,0,0),3)
            cv2.line(im_rgb,(rail2end,0),(rail2end,1271),(255,0,0),3) 
        
        #+str(countplot)+#'edited.png'
        cv2.imwrite( "./opencvresults/group0/"+str(countplot)+'edited.png',im_rgb)
        #cv2.imshow('rail',im_rgb)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        f = plt.figure()
        plt.plot(convResult)
        plt.ylim(ymax =  min(convResult)*2, ymin = min(convResult)/2)
        plt.plot([rail1start,rail1end],[min(convResult[rail1start:rail1end]), min(convResult[rail1start:rail1end])])
        plt.plot([rail2start,rail2end],[min(convResult[rail2start:rail2end]), min(convResult[rail2start:rail2end])])
        #plt.axis([0, 600, min(convResult), max(convResult)])
        # Right Y-axis labels
        plt.text(1.02, 0.5, r"$g(j)$", {'color': 'k', 'fontsize': 10},
                 horizontalalignment='left',
                 verticalalignment='center',
                 rotation=90,
                 clip_on=False,
                 transform=plt.gca().transAxes)
    
        plt.xlabel('j')
        #plt.xticks((1,2), ('group 0', 'group 1'), color='k', size=10)
        plt.show()
    
        f.savefig("./opencvresults/group0/img_segmentation"+str(countplot)+'.png', bbox_inches='tight')
    
    
    #plot code
    if False:
        yLPlot=Image.fromarray(255-yL)
        yLPlot
    
        yRPlot=Image.fromarray(255-yR)
        yRPlot


###############################################################################
#loop over group 1    
countplot=0 
for filename in glob.glob('./Images/Examplesof1/*.jpg'): #assuming gif
    im = cv2.imread(filename,0)
    #im=Image.open(filename)
    image_list_1.append(im)
    
     # Segmentation
    imgCutL, imgCutR, rail1start, rail1end, rail2start, rail2end, convResult = ImageSegmentation(im,railwidth)
    
    #get the results from the correct rail
    std_column_value, railside = RailResults(imgCutL,imgCutL)
    
    
    if method == "MaxstdColumn":
        #no difference but this is consistent with the code
        yL = std_column_value
        yR = std_column_value
    
    if method == "Custom":    
        # Filtering method      
        yL = CustomFilteringmethod(imgCutL)
        yR = CustomFilteringmethod(imgCutR)
    
    if method == "stdFilter":    
        # Filtering method      
        yL = stdFilter(imgCutL)
        yR = stdFilter(imgCutR)

    
    if False:       
        from statsmodels.graphics.gofplots import qqplot
        from scipy.stats import shapiro
        from scipy.stats import normaltest
        from scipy import stats
        
        for i in range(imgCutL.shape[1]):
            #plt.hist(imgCutL[:,i])
            #plt.show()
            #stat, p = shapiro(imgCutL[:,i])
            stat, p = normaltest(imgCutL[:,i])
            kolmosmirnoff=stats.kstest(imgCutL[:,i], 'norm')
            print(kolmosmirnoff)
            #print(p)
            qqplot(imgCutL[:,i], line='s')
            plt.show()
    
    if not method=='MaxstdColumn':
        plt.subplot(1, 2, 1)#+countplot)
        plt.imshow(yL,cmap='Greys')
        plt.title('left rail, group 1')
        plt.subplot(1, 2, 2)#+countplot)
        plt.title('right rail, group 1')
        plt.imshow(yR, cmap='Greys')
        
        plt.show()
    
    countplot+=1
    
   
    #append results
    #left rail
    if railside==0:
        max_value=np.max(yL)
        mean_value=np.mean(yL)
        std_value=np.std(yL)
        #add values to the summary list
        summary_list_1.append([max_value, mean_value, std_value])
    
    #right rail
    if railside==1:
        max_value=np.max(yR)
        mean_value=np.mean(yR)
        std_value=np.std(yR)
        #add values to the summary list
        summary_list_1.append([max_value, mean_value, std_value])
    
    
    if not method== 'stdFilter':
        im_rgb = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
        cv2.imwrite( "./opencvresults/group1/"+str(countplot)+'normal.png',im_rgb)
        
        #
        im_rgb[:,:,:]=0
        #flip black and white???
        #draw faults with red 
        im_rgb[0:im_rgb.shape[0],rail1start:rail1end,0] = yL
        im_rgb[0:im_rgb.shape[0],rail1start:rail1end,1] = yL
        im_rgb[0:im_rgb.shape[0],rail1start:rail1end,2] = yL
    
        im_rgb[0:im_rgb.shape[0],rail2start:rail2end,0] = yR
        im_rgb[0:im_rgb.shape[0],rail2start:rail2end,1] = yR
        im_rgb[0:im_rgb.shape[0],rail2start:rail2end,2] = yR
        
        im_rgb = cv2.bitwise_not(im_rgb)
        
        if railside==0: 
            cv2.line(im_rgb,(rail1start,0),(rail1start,1271),(255,0,0),3)
            cv2.line(im_rgb,(rail1end,0),(rail1end,1271),(255,0,0),3)
        
            cv2.line(im_rgb,(rail2start,0),(rail2start,1271),(255,0,0),2)
            cv2.line(im_rgb,(rail2end,0),(rail2end,1271),(255,0,0),2)
            
        if railside==1: 
            cv2.line(im_rgb,(rail1start,0),(rail1start,1271),(255,0,0),2)
            cv2.line(im_rgb,(rail1end,0),(rail1end,1271),(255,0,0),2)
        
            cv2.line(im_rgb,(rail2start,0),(rail2start,1271),(255,0,0),3)
            cv2.line(im_rgb,(rail2end,0),(rail2end,1271),(255,0,0),3)    
        
        
        #+str(countplot)+#'edited.png'
        cv2.imwrite( "./opencvresults/group1/"+str(countplot)+'edited.png',im_rgb)
        
                
        f = plt.figure()
        plt.plot(convResult)
        plt.ylim(ymax =  min(convResult)*2, ymin = min(convResult)/2)
        plt.plot([rail1start,rail1end],[min(convResult[rail1start:rail1end]), min(convResult[rail1start:rail1end])])
        plt.plot([rail2start,rail2end],[min(convResult[rail2start:rail2end]), min(convResult[rail2start:rail2end])])
        #plt.axis([0, 600, min(convResult), max(convResult)])
        # Right Y-axis labels
        plt.text(1.02, 0.5, r"$g(j)$", {'color': 'k', 'fontsize': 10},
                 horizontalalignment='left',
                 verticalalignment='center',
                 rotation=90,
                 clip_on=False,
                 transform=plt.gca().transAxes)
    
        plt.xlabel('j')
        #plt.xticks((1,2), ('group 0', 'group 1'), color='k', size=10)
        plt.show()
    
        f.savefig("./opencvresults/group1/img_segmentation"+str(countplot)+'.png', bbox_inches='tight')
    


#%%

#Show results 
summary_0 = np.asarray(summary_list_0)    
summary_1 = np.asarray(summary_list_1)    

#max_0 = max(summary_0[:,0])  
#max_1 = max(summary_1[:,0])    
max_0_of_max = np.max(summary_0[:,0])
mean_0_of_max = np.mean(summary_0[:,0])  
std_0_of_max = np.std(summary_0[:,0])  


max_1_of_max = np.max(summary_1[:,0])  
mean_1_of_max = np.mean(summary_1[:,0])    
std_1_of_max = np.std(summary_1[:,0])    

 
print(max_0_of_max, " & ", mean_0_of_max, " & ", std_0_of_max, " & ", max_1_of_max, " & ", mean_1_of_max, " & ", std_1_of_max)


#change print 
#print("max 0:", max_0, ", max 1:", max_1, ", mean 0:", mean_0, ", mean 1:", mean_1, ", std 0:", std_0, ", std 1:", std_1)

maximum = [summary_0[:,0],
     summary_1[:,0]]
plt.boxplot(maximum)
plt.text(1.02, 0.5, r"$output_{max}$", {'color': 'k', 'fontsize': 10},
         horizontalalignment='left',
         verticalalignment='center',
         rotation=90,
         clip_on=False,
         transform=plt.gca().transAxes)
plt.xticks((1,2), ('group 0', 'group 1'), color='k', size=10)
plt.show()



#
mean_0_of_mean = np.mean(summary_0[:,1])  
mean_1_of_mean = np.mean(summary_1[:,1])    

std_0_of_mean = np.std(summary_0[:,1])  
std_1_of_mean = np.std(summary_1[:,1])    

mean_0_of_std = np.mean(summary_0[:,2])  
mean_1_of_std = np.mean(summary_1[:,2])    

std_0_of_std = np.std(summary_0[:,2])  
std_1_of_std = np.std(summary_1[:,2])   


mean = [summary_0[:,1],
     summary_1[:,1]]
plt.boxplot(mean)
# Right Y-axis labels
plt.text(1.02, 0.5, r"$\mu_{max}$", {'color': 'k', 'fontsize': 10},
         horizontalalignment='left',
         verticalalignment='center',
         rotation=90,
         clip_on=False,
         transform=plt.gca().transAxes)
plt.xticks((1,2), ('group 0', 'group 1'), color='k', size=10)
plt.show()


std = [summary_0[:,2],
     summary_1[:,2]]

plt.boxplot(std)
# Right Y-axis labels
plt.text(1.02, 0.5, r"$std_{max}$", {'color': 'k', 'fontsize': 10},
         horizontalalignment='left',
         verticalalignment='center',
         rotation=90,
         clip_on=False,
         transform=plt.gca().transAxes)
plt.xticks((1,2), ('group 0', 'group 1'), color='k', size=10)
plt.show()

#import pandas as pd
#import numpy as np
#df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
#df.plot.box(grid='True')
 
    
# example data
#mu = 100 # mean of distribution
#sigma = 15 # standard deviation of distribution
#x = mu + sigma * np.random.randn(10000)
 
#box plots.


#histogram
#num_bins = 5
# the histogram of the data
#n, bins, patches = plt.hist(summary_0[:,0], num_bins, normed=1, facecolor='blue', alpha=0.5)
 
# add a 'best fit' line
#y = mlab.normpdf(bins, mean_0_of_mean, std_0_of_mean)
#plt.plot(bins, y, 'r--')
#plt.xlabel('Smarts')
#plt.ylabel('Probability')
#plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
 
# Tweak spacing to prevent clipping of ylabel
#plt.subplots_adjust(left=0.15)
#plt.show()

#Normalization group 0
#summary_0[:,0] = (summary_0[:,0]-max_0)/(max_1-max_0)


#Normalization group 1
#summary_1[:,0] = (summary_1[:,0]-max_0)/(max_1-max_0)    

   
 

#%% Delete this later
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