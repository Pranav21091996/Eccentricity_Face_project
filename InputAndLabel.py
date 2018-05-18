import os
import cv2
import numpy as np

def normalize(x):

	a = 0.
	b = 1.
	min = 0
	max = 255

	return a + (((x - min)*(b - a))/(max - min))

def crop_center_resize(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    img = img[starty:starty+cropy,startx:startx+cropx,:]	
    return cv2.resize(img, (32,32))
num_crops = 4
dataSet_directory = 'Dataset100Class/Cropped'
InputImage = []
Labels = []
i=0
for ImageDir in os.listdir(dataSet_directory):
	index=(32,64,128,256)
	#im = cv2.imread(dataSet_directory+'/'+ImageDir)
	#print(dataSet_directory+ImageDir)
	#im  = normalize(im)
	#InputImage.append(im)
	image_directory = dataSet_directory+'/'+ImageDir
	for image in os.listdir(image_directory):
		list = []
		im = cv2.imread(image_directory+'/'+image)
		im = cv2.resize(im, (256,256))
		for crop_index in range(num_crops):
                    
                    imageToCrop=im
                    resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
                    resizedImage= normalize(resizedImage)
                    list.append(resizedImage)
		InputImage.append(list)	                    
        
		#im  = normalize(im)
		#InputImage.append(im)
		Labels.append(i)
	i+=1	
	print(i)
	#if 'bird' in ImageDir:
	#	Labels.append(0)
	#elif 'butterfly' in ImageDir:
	#	Labels.append(1)
	#elif 'car'	in ImageDir:
	#	Labels.append(2)
	#elif 'dog'	in ImageDir:
	#	Labels.append(3)
	#elif 'fruit' in ImageDir:
	#	Labels.append(4)
	#elif 'horse' in ImageDir:
	#	Labels.append(5)
	#elif 'frog' in ImageDir:
	#	Labels.append(6)
	#elif 'jet' in ImageDir:
	#	Labels.append(7)
	#elif 'elephant' in ImageDir:
		#Labels.append(8)
	#elif 'tree' in ImageDir:
		#Labels.append(9)
	#else:
	#	continue 
print(np.shape(InputImage))
np.save('Class100EccentricityCropped',InputImage)
np.save('Class100EccentricityCroppedLabel',Labels)

		
		
	  