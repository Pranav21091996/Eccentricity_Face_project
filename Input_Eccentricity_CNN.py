import matplotlib.pyplot as plt 
import numpy as np 
import os
import cv2
from scipy.misc import toimage
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random


num_channels = 3
img_size = 32
num_crops = 4

def loadDataSet():
	Data = np.load('/home/pranav/Desktop/SUTD/PaddedInput.npy')
	Labels = np.load('/home/pranav/Desktop/SUTD/PaddedInputLabels.npy')
	return Data, Labels

def normalize(x):
	a = 0.
	b = 1.
	min = 0
	max = 255
	return a + (((x - min)*(b - a))/(max - min))

def one_hot_encode(Labels):
	encoder = preprocessing.LabelBinarizer()
	encoder.fit([0,1,2,3,4,5,6,7,8,9])
	x = encoder.transform(Labels)

	return x

def resizePadImage(image):
	image = cv2.resize(image, (20,20)) 
	black = [0,0,0]
	padded = cv2.copyMakeBorder(image, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=black)
	return padded

def paddedresizedImage(Data):
	paddedImage = []
	for images in Data:
		paddedImage.append(resizePadImage(images))
	return paddedImage

def crop_center_resize(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    img = img[starty:starty+cropy,startx:startx+cropx,:]	
    return cv2.resize(img, (32,32))

def showImage(image):
	cv2.imshow('image', image)	
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def _convert_images(Data):
    
    index=(32,64,128,256)
    allStackedImages=[]
     
    for imageIndex in range(len(Data)):
              
                list=[]
                for crop_index in range(num_crops):
                    
                    imageToCrop=Data[imageIndex]
                    resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
                    resizedImage= normalize(resizedImage)
                    list.append(resizedImage)
                    
                stackedImage=np.vstack((list))
                stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
                allStackedImages.append(stackedImage)
               
    return allStackedImages


Data, Labels = loadDataSet()
#padded_Data = paddedresizedImage(Data)
cropped_stacked_input = _convert_images(Data)
np.save('PaddedCentered_stacked_input_256*256', cropped_stacked_input)
print(np.shape(cropped_stacked_input))


