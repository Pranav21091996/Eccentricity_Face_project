import numpy as np 
import cv2
import matplotlib.pyplot as plt 
img = np.load("/home/pranav/Desktop/SUTD/PaddedCentered_stacked_input_256*256.npy")
plt.subplot(141),plt.imshow(img[2000][0])
plt.subplot(142),plt.imshow(img[2000][1])
plt.subplot(143),plt.imshow(img[2000][2])
plt.subplot(144),plt.imshow(img[2000][3])
plt.show()
