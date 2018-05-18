import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random

TRAINING_EPOCHS = 3000
BATCH_SIZE = 100
DISPLAY_STEP = 10

def one_hot_encode(x):
	encoder = preprocessing.LabelBinarizer()
    list1 = np.arange(0,100)
	encoder.fit(list1)
	x = encoder.transform(x)
	return x

Data = np.load('NormalCNNImage.npy')
Labels = np.load('NormalCNNLabels.npy')
one_hot_labels = one_hot_encode(Labels)

print("size of the dataset", len(Data))
print("shape of the dataset", np.shape(Data))
train_Data, test_Data, train_label,test_Label = train_test_split(Data, one_hot_labels, test_size = 0.2)
train_Data, valid_features, train_label,valid_Label = train_test_split(train_Data, train_label, test_size = 0.2)
print("size of the training data", len(train_Data))
print("size of the test data", len(test_Data))
print("size of the Validation data", len(valid_features))

VALIDATION_SIZE = len(valid_features)

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
        train_Data = train_Data[perm]
        train_label = train_label[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_Data[start:end], train_label[start:end]

def neural_net_image_input():

	x = tf.placeholder(tf.float32, shape = [None, 32, 32, 3], name = 'x')
	return x

def neural_net_label_input(n_classes):
	
	y = tf.placeholder(tf.float32, shape = [None, n_classes], name = 'y')
	return y	

def neural_net_keep_prob_input():

	keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
	return keep_prob

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    
    F_W = tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1], x_tensor.get_shape().as_list()[3], conv_num_outputs], stddev=0.05, mean=0.0))
    F_b = tf.Variable(tf.zeros(conv_num_outputs))
    
    layer1 = tf.nn.conv2d(x_tensor, 
                          F_W, 
                          strides=[1, conv_strides[0], conv_strides[1], 1], 
                          padding = 'SAME')
    layer2a = tf.nn.bias_add(layer1, F_b)
    layer2b = tf.nn.relu(layer2a)
    layer2c = tf.nn.max_pool(
                layer2b,
                ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                strides=[1, pool_strides[0], pool_strides[1], 1], 
                padding = 'SAME')
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
    conv_ksize = (3,3)
    conv_strides = (1,1)
    pool_ksize = (2,2)
    pool_strides = (2,2)
  
    conv_num_outputs = 24
    conv = conv2d_maxpool(x_tensor,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides)

    conv_num_outputs = 48
    conv2 = conv2d_maxpool(conv,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides)

    conv_num_outputs = 96
    conv3 = conv2d_maxpool(conv2,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides)


    conv_num_outputs = 192
    conv4 = conv2d_maxpool(conv3,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides)


    conv_num_outputs = 384
    conv5 = conv2d_maxpool(conv4,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides)

    flat = flatten(conv5)

    fc1 = fully_conn(flat,512)
    fc1 = tf.nn.dropout(fc1,keep_prob)
    fc2 = fully_conn(flat,512)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    out = output(fc2, 8)

    return out

x = neural_net_image_input()
y = neural_net_label_input(8)    
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
num_examples = train_Data.shape[0]
print(num_examples)

# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)

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
            validation_accuracy = accuracy.eval(feed_dict={ x: valid_features[0:BATCH_SIZE],y: valid_Label[0:BATCH_SIZE],keep_prob: 0.5})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
         else:
            print('train_accuracicy => %.4f for step %d'%(train_accuracy, i))
            train_accuracies.append(train_accuracy)
         if i%(DISPLAY_STEP*10) == 0 and i:
            DISPLAY_STEP *=10

    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,keep_prob: 0.5})
# check final accuracy on validation set  
if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={x: valid_features, y: valid_Label,keep_prob: 0.5})
    print('validation_accuracy => %.4f'%validation_accuracy)  


                                             
test_accuracy = accuracy.eval(feed_dict={x: test_Data, 
                                         y: test_Label,
                                         keep_prob: 0.5})
print('test_accuracy => %.4f'%test_accuracy)  
best1,best2 = sess.run([y,Y_pred],feed_dict={ x: test_Data,y: test_Label,keep_prob: 0.5})
for i in range(len(best1)):
    print(best1[i])
    print(best2[i])


