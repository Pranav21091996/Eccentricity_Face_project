import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2
import matplotlib.pyplot as plt

def normalize(x):

    a = 0.
    b = 1.
    min = 0
    max = 255

    return a + (((x - min)*(b - a))/(max - min))

image = []
dataSet_directory = 'Dataset100Class/Original/n00007846/'
for ImageDir in os.listdir(dataSet_directory):
	im = cv2.imread(os.path.join(dataSet_directory,ImageDir))
	im = cv2.resize(im,(64,64))
	im = normalize(im)
	image.append(im)

meanImg = np.mean(image, axis = 0)
norm_image = []
for im in image:
	im = im - meanImg
	norm_image.append(im)


