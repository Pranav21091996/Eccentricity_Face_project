import matplotlib.pyplot as plt 
#train_accuracy = [10,13,16,16,14,14,20,18,13,17,22,18,29,19,32,34,28,41,31,41,59,60,59,73,72,86,86,82,90,97,100] 
step = [0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000]
#validation_accuracy = [6,14,19,18,16,13,9,14,15,16,17,17,20,24,29,28,32,33,34,41,44,51,53,52,52,62,51,60,57,62,59]
NormalCNN = [14,16,15,19,15,15,16,20,17,15,23,18,22,21,26,31,35,50,33,44,56,59,64,59,70,82,76,86,91,99,100]
FullImageECNN = [18,14,11,12,16,14,17,17,16,23,11,21,26,22,27,39,46,35,43,46,64,74,63,66,76,82,82,84,83,99,100]
BoundBoxECNN = [12,12,15,14,13,16,10,17,18,17,26,32,26,30,32,36,37,36,41,41,55,57,63,76,71,85,85,95,95,100,100]
BoundBoxCenter = [10,13,16,16,14,14,20,18,13,17,22,18,29,19,32,34,28,41,31,41,59,60,59,73,72,86,86,82,90,97,100]
plt.plot(step,NormalCNN,'r')
plt.plot(step,FullImageECNN,'b')
plt.plot(step,BoundBoxECNN,'g')
plt.plot(step,BoundBoxCenter,'y')
#plt.plot(step,train_accuracy,'r')
#plt.plot(step,validation_accuracy,'b')
#plt.gca().legend(('train_accuracy','validation_accuracy'))
#plt.title('Training Accuracy vs Validation_accuracy')
plt.gca().legend(('NormalCNN','FullImageECNN','BoundBoxECNN','BoundBoxCenter'))
plt.title('Training Accuracy of NormalCNN,FullImageECNN,BoundBoxECNN,BoundBoxCenter')
plt.xlabel('Step_Size')
plt.ylabel('Accuracy')
plt.show()

