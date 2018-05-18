import numpy as np 
fp = open('pred')
lines = fp.readlines()
labels = []
prediction = []
count = 0
def topNAccuracy(sorted_labels,sorted_predictions,N):
	correct = 0
	for i in range(len(sorted_predictions)):
		for j in range(N):
			if(sorted_labels[i][0]==sorted_predictions[i][j]):
				correct+=1
				break
	return correct			
for line in lines:
	list = []
	for x in line.split():
		list.append(float(x))
	if(count%2==0):
		labels.append(list)
	else:
		prediction.append(list)	
	count+=1
sorted_labels = []	
sorted_predictions = []
for i in range(len(labels)):	
	sorted_labels.append(np.argsort(labels[i])[::-1][:len(labels[0])])
	sorted_predictions.append(np.argsort(prediction[i])[::-1][:len(prediction[0])])
Accuracy = topNAccuracy(sorted_labels,sorted_predictions,1)
print(Accuracy/len(sorted_predictions))
Accuracy = topNAccuracy(sorted_labels,sorted_predictions,2)
print(Accuracy/len(sorted_predictions))
Accuracy = topNAccuracy(sorted_labels,sorted_predictions,3)
print(Accuracy/len(sorted_predictions))
Accuracy = topNAccuracy(sorted_labels,sorted_predictions,4)
print(Accuracy/len(sorted_predictions))
Accuracy = topNAccuracy(sorted_labels,sorted_predictions,5)
print(Accuracy/len(sorted_predictions))



