import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
import random
import os
import numpy as np
import matplotlib.pyplot as plt 
import cv2

def normalize(x):

    a = 0.
    b = 1.
    min = 0
    max = 255

    return a + (((x - min)*(b - a))/(max - min))



dataSet_directory = 'NewFaceCentered64*64'

testData_directory2 ='64_8pix_down'
testData_directory3 =  '64_8pix_up'
testData_directory4 = '64_8pix_left'
testData_directory5 = '64_8pix_right'
'''
testData_directory6 = 
testData_directory7 = '16pixrightdown128*128'
testData_directory8 ='16pixrightup128*128'
testData_directory9 = 
'''
InputImage = []
Labels = []
test_Data1 = []
test_label1 = []

test_Data2 = []
test_label2 = []
test_Data3 = []
test_label3 = []
test_Data4 = []
test_label4 = []
test_Data5 = []
test_label5 = []
'''
test_Data6 = []
test_label6 = []
test_Data7 = []
test_label7 = []
test_Data8 = []
test_label8 = []
test_Data9 = []
test_label9 = []
'''
num_crops = 4
img_size = 64
i=0

TRAINING_EPOCHS = 10000
BATCH_SIZE = 128
DISPLAY_STEP = 10

def crop_center_resize(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    img = img[starty:starty+cropy,startx:startx+cropx,:]	
    return cv2.resize(img, (64,64))

print(dataSet_directory)
for ImageDir in os.listdir(dataSet_directory):
	index=(32,64,128,256)
	image_directory = dataSet_directory+'/'+ImageDir
	
	test_image_directory2 = testData_directory2+'/'+ImageDir
	test_image_directory3 = testData_directory3+'/'+ImageDir
	test_image_directory4 = testData_directory4+'/'+ImageDir
	test_image_directory5 = testData_directory5+'/'+ImageDir
	'''
	test_image_directory6 = testData_directory6+'/'+ImageDir
	test_image_directory7 = testData_directory7+'/'+ImageDir
	test_image_directory8 = testData_directory8+'/'+ImageDir
	test_image_directory9 = testData_directory9+'/'+ImageDir
	'''
	print(image_directory)
	print(i)
	count=0
	for image in os.listdir(image_directory):
		list = []
		list1 = []
		try:
			im = cv2.imread(image_directory+'/'+image)
			im = cv2.resize(im, (256,256))

			for crop_index in range(num_crops):
				imageToCrop=im
				resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
				resizedImage= normalize(resizedImage)
				list.append(resizedImage)
			stackedImage=np.vstack((list))
			stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
			count+=1
			if(count<100):
				InputImage.append(stackedImage)
				Labels.append(i)
			
			else:
				test_Data1.append(stackedImage)
				test_label1.append(i)
				
				im2 = cv2.imread(test_image_directory2+'/'+image)
				im2 = cv2.resize(im2, (256,256))
				for crop_index in range(num_crops):
					imageToCrop=im2
					resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
					resizedImage= normalize(resizedImage)
					list1.append(resizedImage)
				stackedImage=np.vstack((list1))
				list1 = []
				stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
				test_Data2.append(stackedImage)
				test_label2.append(i)

				im3 = cv2.imread(test_image_directory3+'/'+image)
				im3 = cv2.resize(im3, (256,256))
				for crop_index in range(num_crops):
					imageToCrop=im3
					resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
					resizedImage= normalize(resizedImage)
					list1.append(resizedImage)
				stackedImage=np.vstack((list1))
				list1 = []
				stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
				test_Data3.append(stackedImage)
				test_label3.append(i)

				im4 = cv2.imread(test_image_directory4+'/'+image)
				im4 = cv2.resize(im4, (256,256))
				for crop_index in range(num_crops):
					imageToCrop=im4
					resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
					resizedImage= normalize(resizedImage)
					list1.append(resizedImage)
				stackedImage=np.vstack((list1))
				list1 = []
				stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
				test_Data4.append(stackedImage)
				test_label4.append(i)
				
				im5 = cv2.imread(test_image_directory5+'/'+image)
				im5 = cv2.resize(im5, (256,256))
				for crop_index in range(num_crops):
					imageToCrop=im5
					resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
					resizedImage= normalize(resizedImage)
					list1.append(resizedImage)
				stackedImage=np.vstack((list1))
				list1 = []
				stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
				test_Data5.append(stackedImage)
				test_label5.append(i)
				'''
				im6 = cv2.imread(test_image_directory6+'/'+image)
				im6 = cv2.resize(im6, (256,256))
				for crop_index in range(num_crops):
					imageToCrop=im6
					resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
					resizedImage= normalize(resizedImage)
					list1.append(resizedImage)
				stackedImage=np.vstack((list1))
				list1 = []
				stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
				test_Data6.append(stackedImage)
				test_label6.append(i)
				
				im7 = cv2.imread(test_image_directory7+'/'+image)
				im7 = cv2.resize(im7, (256,256))
				for crop_index in range(num_crops):
					imageToCrop=im7
					resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
					resizedImage= normalize(resizedImage)
					list1.append(resizedImage)
				stackedImage=np.vstack((list1))
				list1 = []
				stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
				test_Data7.append(stackedImage)
				test_label7.append(i)
				
				im8 = cv2.imread(test_image_directory8+'/'+image)
				im8 = cv2.resize(im8, (256,256))
				for crop_index in range(num_crops):
					imageToCrop=im8
					resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
					resizedImage= normalize(resizedImage)
					list1.append(resizedImage)
				stackedImage=np.vstack((list1))
				list1 = []
				stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
				test_Data8.append(stackedImage)
				test_label8.append(i)
				
				im9 = cv2.imread(test_image_directory9+'/'+image)
				im9 = cv2.resize(im9, (256,256))
				for crop_index in range(num_crops):
					imageToCrop=im9
					resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
					resizedImage= normalize(resizedImage)
					list1.append(resizedImage)
				stackedImage=np.vstack((list1))
				list1 = []
				stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
				test_Data9.append(stackedImage)
				test_label9.append(i)
				'''
		except Exception as e:
			print ("Unexpected error:", str(e))					
	i+=1	


def one_hot_encode(x):
	encoder = preprocessing.LabelBinarizer()
	list1 = np.arange(0,100)
	encoder.fit(list1)
	x = encoder.transform(x)
	return x



Data = InputImage
Labels = Labels
one_hot_labels = one_hot_encode(Labels)
#train_Data, test_Data1, train_label, test_label1 = train_test_split(Data, one_hot_labels, test_size=0.2)
one_hot_labels_test1 = one_hot_encode(test_label1)
one_hot_labels_test2 = one_hot_encode(test_label2)
one_hot_labels_test3 = one_hot_encode(test_label3)
one_hot_labels_test4 = one_hot_encode(test_label4)
one_hot_labels_test5 = one_hot_encode(test_label5)
'''
one_hot_labels_test6 = one_hot_encode(test_label6)
one_hot_labels_test7 = one_hot_encode(test_label7)
one_hot_labels_test8 = one_hot_encode(test_label8)
one_hot_labels_test9 = one_hot_encode(test_label9)
'''

train_Data,train_label = shuffle(Data,one_hot_labels,random_state=20)
test_Data1,test_label1 = shuffle(test_Data1,one_hot_labels_test1,random_state=20)
test_Data2,test_label2 = shuffle(test_Data2,one_hot_labels_test2,random_state=20)
test_Data3,test_label3 = shuffle(test_Data3,one_hot_labels_test3,random_state=20)
test_Data4,test_label4 = shuffle(test_Data4,one_hot_labels_test4,random_state=20)
test_Data5,test_label5 = shuffle(test_Data5,one_hot_labels_test5,random_state=20)
'''
test_Data6,test_label6 = shuffle(test_Data6,one_hot_labels_test6)
test_Data7,test_label7 = shuffle(test_Data7,one_hot_labels_test7)
test_Data8,test_label8 = shuffle(test_Data8,one_hot_labels_test8)
test_Data9,test_label9 = shuffle(test_Data9,one_hot_labels_test9)
'''

print("size of the train data", len(Data))
print("shape of the train data", np.shape(Data))
print("size of the test data1", len(test_Data1))
print("size of the test data2", len(test_Data2))
print("size of the test data3", len(test_Data3))
print("size of the test data4", len(test_Data4))
print("size of the test data5", len(test_Data5))
'''
print("size of the test data2", len(test_Data6))
print("size of the test data3", len(test_Data7))
print("size of the test data4", len(test_Data8))
print("size of the test data5", len(test_Data9))
'''
VALIDATION_SIZE = len(test_Data1)

def next_batch(batch_size):    
	
	global train_Data
	global train_label
	global index_in_epoch
	global epochs_completed
	
	start = index_in_epoch
	index_in_epoch += batch_size
	
	# when all trainig data have been already used, it is reorder randomly    
	if index_in_epoch > num_examples:
		# finished epoch
		epochs_completed += 1
		# shuffle the data
		perm = np.arange(num_examples)
		np.random.shuffle(perm)
		train_Data = [train_Data[i] for i in perm]
		train_label = [train_label[i] for i in perm]
		# start next epoch
		start = 0
		index_in_epoch = batch_size
		assert batch_size <= num_examples
	end = index_in_epoch
	return train_Data[start:end], train_label[start:end]
def neural_net_image_input():

	x = tf.placeholder(tf.float32, shape = [None, 4,64,64,3], name = 'x')
	return x

def neural_net_label_input(n_classes):
	
	y = tf.placeholder(tf.float32, shape = [None, n_classes], name = 'y')
	return y	

def neural_net_keep_prob_input():

	keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
	return keep_prob

def conv3d(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides,in_channels):
	
	if(x_tensor.get_shape()[2] < 5 ):
		F_W = tf.Variable(tf.truncated_normal([1,1,1, in_channels,  conv_num_outputs], stddev=0.05, mean=0.0))
	else:
		F_W = tf.Variable(tf.truncated_normal([1,conv_ksize[0], conv_ksize[1], in_channels,  conv_num_outputs], stddev=0.05, mean=0.0))	
	F_b = tf.Variable(tf.zeros(conv_num_outputs))
	
	layer1 = tf.nn.conv3d(x_tensor, 
						  F_W, 
						  strides=[1, 1, 1, 1, 1], 
						  padding = 'VALID')
	layer2a = tf.nn.bias_add(layer1, F_b)
	layer2b = tf.nn.relu(layer2a)
	return layer2b

def maxpool3d(conv,pool_ksize,pool_strides,flag):
	if(flag == 0):
		layer2c = tf.nn.max_pool3d(
				conv,
				ksize=[1, 1, pool_ksize[0], pool_ksize[1], 1],
				strides=[1, 1, pool_strides[0], pool_strides[1], 1], 
				padding = 'VALID')
	elif(flag == 1):
		kernel_size = conv.get_shape()[2]
		layer2c = tf.nn.max_pool3d(
				conv,
				ksize=[1, 1, kernel_size, kernel_size, 1],
				strides=[1, 1, pool_strides[0], pool_strides[1], 1], 
				padding = 'VALID')
	else:
		layer2c = tf.nn.max_pool3d(
				conv,
				ksize=[1, 4, 1, 1, 1],
				strides=[1, 1, 1, 1, 1], 
				padding = 'VALID')
		
	return layer2c
	

def flatten(x_tensor):

	shape = x_tensor.get_shape().as_list()
	dim = np.prod(shape[1:])
	x_tensor_flat = tf.reshape(x_tensor, [-1, dim])

	return x_tensor_flat

def fully_conn(x_tensor, num_outputs):

	inputs = x_tensor.get_shape().as_list()[1]
	weights = tf.Variable(tf.truncated_normal([inputs, num_outputs], stddev = 0.05, mean = 0.0))
	bias = tf.Variable(tf.zeros(num_outputs))
	logits = tf.add(tf.matmul(x_tensor,weights), bias)

	return tf.nn.relu(logits)

def output(x_tensor, num_outputs):

	inputs = x_tensor.get_shape().as_list()[1]
	weights = tf.Variable(tf.truncated_normal([inputs, num_outputs], stddev = 0.05, mean = 0.0))
	bias = tf.Variable(tf.zeros(num_outputs))
	logits = tf.add(tf.matmul(x_tensor,weights), bias)

	return logits

def conv_net(x, keep_prob):

	x_tensor = x
	conv_ksize = (5,5)
	conv_strides = (1,1)
	pool_ksize = (3,3)
	pool_strides = (1,1)

	flag = 0
	conv_num_outputs = 192
	in_channels = 3
	conv = conv3d(x_tensor,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv = maxpool3d(conv,pool_ksize,pool_strides,flag)

	conv_num_outputs = 192
	in_channels = 192
	conv2 = conv3d(conv,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv2 = maxpool3d(conv2,pool_ksize,pool_strides,flag)

	conv_num_outputs = 192
	in_channels = 192
	conv3 = conv3d(conv2,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv3 = maxpool3d(conv3,pool_ksize,pool_strides,flag)

	conv_num_outputs = 192
	in_channels = 192
	conv4 = conv3d(conv3,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv4 = maxpool3d(conv4,pool_ksize,pool_strides,flag)

	conv_num_outputs = 256
	in_channels = 192
	flag = 1
	conv5 = conv3d(conv4,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv5 = maxpool3d(conv5,pool_ksize,pool_strides,flag)
	flag = 2
	conv5 = maxpool3d(conv5,pool_ksize,pool_strides,flag)


	flat = flatten(conv5)

	fc1 = fully_conn(flat,512)
	fc1 = tf.nn.dropout(fc1,keep_prob)
#	fc2 = fully_conn(flat,512)
#	fc2 = tf.nn.dropout(fc2, keep_prob)

	out = output(fc1, 100)

	return out

x = neural_net_image_input()
y = neural_net_label_input(100)    
keep_prob = neural_net_keep_prob_input()

logits = conv_net(x, keep_prob)
Y_pred = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32), name = 'accuracy')


predict = tf.argmax(Y_pred, 1)

epochs_completed = 0
index_in_epoch = 0
num_examples = len(train_Data)
print(num_examples)

# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)

#saver = tf.train.Saver()
# visualisation variables
train_accuracies = []
validation_accuracies = []

DISPLAY_STEP=1

for i in range(TRAINING_EPOCHS):

    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)  
          

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%DISPLAY_STEP == 0 or (i+1) == TRAINING_EPOCHS:

         train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
                                                  y: batch_ys,
                                                  keep_prob: 0.5}) 
         if(VALIDATION_SIZE):  
            validation_accuracy = accuracy.eval(feed_dict={ x: test_Data1[0:BATCH_SIZE],y: test_label1[0:BATCH_SIZE],keep_prob: 0.5})
            print('training_accuracy => %.2f for step %d'%(train_accuracy , i))
         else:
            print('train_accuracicy => %.4f for step %d'%(train_accuracy, i))
            train_accuracies.append(train_accuracy)
         if i%(DISPLAY_STEP*10) == 0 and i:
            DISPLAY_STEP *=10
    validation_accuracy = accuracy.eval(feed_dict={ x: test_Data1[0:BATCH_SIZE],y: test_label1[0:BATCH_SIZE],keep_prob: 0.5})
    print('validation_accuracy 1 => %.2f for step %d'%(validation_accuracy, i))
    validation_accuracy = accuracy.eval(feed_dict={ x: test_Data2[0:BATCH_SIZE],y: test_label2[0:BATCH_SIZE],keep_prob: 0.5})
    print('validation_accuracy 2 => %.2f for step %d'%(validation_accuracy, i))
    validation_accuracy = accuracy.eval(feed_dict={ x: test_Data3[0:BATCH_SIZE],y: test_label3[0:BATCH_SIZE],keep_prob: 0.5})
    print('validation_accuracy 3 => %.2f for step %d'%(validation_accuracy, i))
    validation_accuracy = accuracy.eval(feed_dict={ x: test_Data4[0:BATCH_SIZE],y: test_label4[0:BATCH_SIZE],keep_prob: 0.5})
    print('validation_accuracy 4 => %.2f for step %d'%(validation_accuracy, i))
    validation_accuracy = accuracy.eval(feed_dict={ x: test_Data5[0:BATCH_SIZE],y: test_label5[0:BATCH_SIZE],keep_prob: 0.5})
    print('validation_accuracy 5 => %.2f for step %d'%(validation_accuracy, i))
    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,keep_prob: 0.5})
#saver.save(sess, save_file


best1,best2 = sess.run([y,Y_pred],feed_dict={ x: test_Data1[0:BATCH_SIZE],y: test_label1[0:BATCH_SIZE],keep_prob: 0.5})

best3,best4 = sess.run([y,Y_pred],feed_dict={ x: test_Data2[0:BATCH_SIZE],y: test_label2[0:BATCH_SIZE],keep_prob: 0.5})
best5,best6 = sess.run([y,Y_pred],feed_dict={ x: test_Data3[0:BATCH_SIZE],y: test_label3[0:BATCH_SIZE],keep_prob: 0.5})
best7,best8 = sess.run([y,Y_pred],feed_dict={ x: test_Data4[0:BATCH_SIZE],y: test_label4[0:BATCH_SIZE],keep_prob: 0.5})
best9,best10 = sess.run([y,Y_pred],feed_dict={ x: test_Data5[0:BATCH_SIZE],y: test_label5[0:BATCH_SIZE],keep_prob: 0.5})
'''
best11,best12 = sess.run([y,Y_pred],feed_dict={ x: test_Data6[0:BATCH_SIZE],y: test_label6[0:BATCH_SIZE],keep_prob: 0.5})
best13,best14 = sess.run([y,Y_pred],feed_dict={ x: test_Data7[0:BATCH_SIZE],y: test_label7[0:BATCH_SIZE],keep_prob: 0.5})
best15,best16 = sess.run([y,Y_pred],feed_dict={ x: test_Data8[0:BATCH_SIZE],y: test_label8[0:BATCH_SIZE],keep_prob: 0.5})
best17,best18 = sess.run([y,Y_pred],feed_dict={ x: test_Data9[0:BATCH_SIZE],y: test_label9[0:BATCH_SIZE],keep_prob: 0.5})
'''



prediction1 = []
prediction2 = []
prediction3 = []
prediction4 = []
prediction5 = []
'''
prediction6 = []
prediction7 = []
prediction8 = []
prediction9 = []

'''
for i in range(len(best1)):
    prediction1.append(best1[i])
    prediction1.append(best2[i])
    prediction2.append(best3[i])
    prediction2.append(best4[i])
    prediction3.append(best5[i])
    prediction3.append(best6[i])
    prediction4.append(best7[i])
    prediction4.append(best8[i])
    prediction5.append(best9[i])
    prediction5.append(best10[i])
    ''' 
    prediction6.append(best11[i])
    prediction6.append(best12[i])
    prediction7.append(best13[i])
    prediction7.append(best14[i])
    prediction8.append(best15[i])
    prediction8.append(best16[i])
    prediction9.append(best17[i])
    prediction9.append(best18[i])
    '''
    
    
    

    #print(best1[i])
    #print(best2[i])

f1 = open('prediction1','w')
f1.write(str(prediction1))
f2 = open('prediction2','w')
f2.write(str(prediction2))
f3 = open('prediction3','w')
f3.write(str(prediction3))
f4 = open('prediction4','w')
f4.write(str(prediction4))
f5 = open('prediction5','w')
f5.write(str(prediction5))
'''
f6 = open('prediction6','w')
f6.write(str(prediction6))
f7 = open('prediction7','w')
f7.write(str(prediction7))
f8 = open('prediction8','w')
f8.write(str(prediction8))
f9 = open('prediction9','w')
f9.write(str(prediction9))

'''
