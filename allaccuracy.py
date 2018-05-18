import matplotlib.pyplot as plt
steps = [0,1,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000,60000,70000,80000,90000,99999]

#Nomal Image NormalCNN 
Train_accuracy_Normal_CNN_Normal_image = [2,1,0,1,2,3,0,0,0,1,0,3,3,2,5,3,1,5,3,1,2,3,4,4,9,13,8,11,9,21,32,39,51,51,65,79,84,75,96,95,95,93,91,93,93,95]
Validation_acuuracy_Normal_CNN_Normal_image = [1,0,1,0,2,3,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,3,1,8,5,5,10,5,8,12,19,18,16,13,15,16,18,15,21,11,21,15,15,17,15,16]
#size of the dataset  50933
#shape of the dataset (50933, 32, 32, 3)
#size of the training data 32596
#size of the test data 10187
#size of the Validation data 8150
#test_accuracy => 18.66%
#Epochs = 100000

#Normal Image Eccentricity CNN

Train_accuracy_Eccentricity_CNN_Normal_image = [1,0,1,1,1,1,0,1,2,2,1,2,4,3,4,2,2,1,2,3,6,4,13,13,10,13,11,14,10,23,29,44,64,67,74,71,91,88,90,95,91,94,89,91,95,95]
Validation_accuracy_Eccentricity_CNN_Normal_image = [3,1,1,3,2,0,1,0,1,1,3,1,0,0,0,0,0,1,2,0,1,4,5,6,10,7,13,12,15,23,29,36,26,28,26,27,25,34,31,28,29,24,24,25,28,24]
#test_accuracy => 19.46%


#Cropped Normal CNN

Train_accuracy_NormalCNN_Cropped_image = [0,1,0,0,2,3,2,2,0,3,2,5,4,3,0,1,1,3,2,6,2,8,8,11,22,16,14,20,17,41,38,60,66,69,83,80,82,84,97,95,92,96,97,95,94,95]
validation_accuracy_NormalCNN_Cropped_image = [1,2,0,2,2,2,3,2,2,3,3,6,3,3,3,4,4,5,4,4,6,8,8,10,11,13,14,18,27,33,31,36,31,32,42,36,35,33,31,33,35,32,34,30,37,32]
#test_accuracy => 28.97%


#Cropped Eccentricity CNN

Train_accuracy_Eccentricity_Cropped_image = [3,1,1,1,1,2,1,3,1,0,0,4,5,4,3,1,6,4,3,3,6,6,8,20,8,16,17,16,18,33,41,52,66,70,71,81,91,88,94,89,96,100,93,96,97,96]
validation_accuracy_Eccentricity_Cropped_image = [1,1,0,0,0,2,1,3,2,1,2,3,5,6,4,4,7,5,4,9,4,6,9,12,15,9,12,11,18,30,32,32,36,29,27,31,34,34,30,29,27,24,25,26,25,27]
#test_accuracy => 31.06%
#Epochs = 100000


#Padded Centered Normal CNN

Train_accuracy_NormalCNN_PaddedCentered_image = [3,1,4,4,2,2,2,4,1,1,3,1,5,1,2,1,5,3,0,0,5,4,4,8,12,7,10,12,19,17,34,47,48,58,77,75,79,83,96,92,98,95,94,96,95,90]
Validation_accuracy_NormalCNN_PaddedCentered_image = [0,2,2,3,2,2,1,2,1,3,7,4,4,4,4,5,7,6,6,8,7,10,7,10,17,13,16,19,23,24,29,25,23,30,29,27,26,28,22,20,21,20,22,28,24,28]
#test_accuracy => 18.04%

#Padded Centered Eccentricity CNN

Train_accuracy_EccentricityCNN_PaddedCentered_image = [1,0,1,0,0,1,2,3,2,2,0,2,0,3,1,1,2,3,2,3,4,5,4,4,11,7,6,6,12,21,29,48,45,51,66,72,84,91,82,88,95,92,93,94,91,97]
Validation_accuracy_EccentricityCNN_PaddedCentered_image = [0,2,1,2,0,3,1,0,1,1,1,1,1,1,0,0,3,3,4,0,3,5,5,10,10,8,10,14,15,17,26,22,25,24,20,19,19,24,31,26,22,18,27,22,25,21]

#test_accuracy => 18.10%

plt.plot(steps,Train_accuracy_NormalCNN_PaddedCentered_image,'r')
plt.plot(steps,Validation_accuracy_NormalCNN_PaddedCentered_image,'b')
plt.plot(steps,Train_accuracy_EccentricityCNN_PaddedCentered_image,'g')
plt.plot(steps,Validation_accuracy_EccentricityCNN_PaddedCentered_image,'y')
plt.gca().legend(('Train_accuracy_NormalCNN_PaddedCentered_image','Validation_accuracy_NormalCNN_PaddedCentered_image','Train_accuracy_Eccentricity_CNN_PaddedeCentered_image','Validation_accuracy_Eccentricity_CNN_PaddedCentered_image'))
plt.title('Training and Validation Accuracy')
plt.xlabel('Step_Size')
plt.ylabel('Accuracy')
plt.show()
	





