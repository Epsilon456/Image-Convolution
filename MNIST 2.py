#Import libraries
import tensorflow as tf
import numpy as np

#Import data from keras datasets as tensors
"""
Dataset is automatically split into a train dataset (60,000) and test dataset (10,000) images.
x_train and x_test are arrays of shape [n,28,28] which represent 28,28 pixel greyscale images.
y_train and y_test represent the index of the labels corresponding to each images. 
"""
(x_train, y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

#Convert arrays from int arrays to float arrays.
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

##########################Data Sets#################################################
#Define the batch size and number of training epochs.
batchSize = 64
epochs = 10000

#Define placeholders
_U = tf.placeholder(tf.float32,shape=[None,28,28]) #Image data
_Z = tf.placeholder(tf.int32,shape=[None]) #Labels

#Define datasets and tie together.
dU = tf.data.Dataset.from_tensor_slices(_U)
dZ = tf.data.Dataset.from_tensor_slices(_Z)
dataset = tf.data.Dataset.zip((dU,dZ))

#Shuffle and repeat if training. Do not shuffle if testing.
trainQ = 0
if trainQ == 0:
    #Shuffle in batches of 300. Prefetch the next two batches before data is pulled.
    Dataset = dataset.batch(batchSize).shuffle(300).prefetch(128).repeat()
else:
    Dataset = dataset.batch(batchSize).repeat()
    
####################################Set iterator###################################
iterator = Dataset.make_initializable_iterator()
U,Z = iterator.get_next()

def He(fanin):
    """He Xavier variance calculation for initialization
    Inputs: 
        fanin - (int) The number of input inputs to a given layer. 
    Returns: An integer corresponding to the needed variance to initialize the trainable variable."""
    return np.sqrt(2/fanin)

def Convolve(inputTensor,width,filters):
    """Adds a convolutional and pooling layer.  The number of input channels of the convolution
    is determined automatically. The pooling layer performs a max pooling operation that reduces
    the resolution of the output by 1/2.
    Inputs:
        inputTensor: [None,pixels,pixels,filters] tensor representing the image at a given layer of the 
            network
        width: (int) The width of the convolutional kernel
        fitlers: The number of output channels for the output of the convolutional opperation.
    Outputs:
        CV - [None,pixels,pixels,filters] The output of the convolutional layer. (This is for 
             information only as it is an intermediate calculation that is not needed outside
             this function).
        Pool - [None,pixels/2,pixels/2,filters] The output of the pooling layer.
    
    """
    #Calculate the numer of input channels given the shape of the input tensor.
    inputSize = int(inputTensor.shape[-1])
    #Calculate the the number of values in the convolutional kernel corresponding to the inputs to 
        #be used in the variance calculation.
    variance = width**2*inputSize
    
    #Initialize kernel and bias variables using He variance initialization.
    WC = tf.Variable(tf.truncated_normal([width,width,inputSize,filters],stddev=He(variance),dtype=tf.float32))
    BC = tf.Variable(tf.truncated_normal((filters,),stddev = He(filters),dtype=tf.float32))
    
    #Perform the convolution (stride of 1) with zero padding. Add a bias to the result.
    CV = tf.nn.conv2d(inputTensor,WC,[1,1,1,1],"SAME")+BC
    #Apply a max  pooling operation (stride of 2) and apply a leaky_relu activation function
        #to the output (with an alpha value  of .01).
    Pool = tf.nn.leaky_relu(tf.nn.pool(CV,[2,2],"MAX","SAME",strides=[2,2]),.01)
    
    return CV,Pool

def Dense(inputTensor,outShape,Input=None,act='leaky_relu'):
    """Applies a Dense layer to the neural network.
    Inputs:
        inputTensor - [None,pixels,pixels,filters] a tensor corresponding to the image to be 
            fed into the fully connected layer.
        outShape - (int) The number of neurons to be added to the dense layer.
        Input - List [width,width,inputChannels] If the tensor is a rank 4 tensor (such as the)
            one that is the output layer of the convolutional network, then the shape of the input 
            tensor must be specified.
        act - (str) a string index for the type of activation function to be used.
    Returns:
        [n,outShape] a tensor which has undergone a fully connected layer.
    """
    #If the input tensor is a rank 4 tensor, the shape is specified.
    if Input == list:
        #Calculate the number of input values to this layer for variance calculations.
        variance = Input[0]*Input[1]*Input[2]
        #Initialize kernel and bias using He initialization.
        W = tf.Variable(tf.truncated_normal(outShape,stddev=He(variance),dtype=tf.float32))
        B = tf.Variable(tf.truncated_normal((Input[-1],),dtype=tf.float32,stddev=He(Input[-1])))
        #Perform the equivalent of a dot product for rank 4 tensors using einsum notation.
        einsum = 'ijkl,jklm->im'
    
    #Does the same as the above section, except uses a rank 2 tenssor as input.
    else:
        inShape = int(inputTensor.shape[-1])
        W = tf.Variable(tf.truncated_normal([inShape,outShape],stddev=He(inShape),dtype=tf.float32))
        B = tf.Variable(tf.truncated_normal((outShape,),dtype=tf.float32,stddev=He(outShape)))
        einsum = 'ij,jk->ik'
    
    #Applies either a leaky_activation function if requested.
    if act == 'leaky_relu':
        return tf.nn.leaky_relu(tf.einsum(einsum,inputTensor,W)+B,.01)
    elif act == 'linear':
        return tf.einsum(einsum,inputTensor,W)+B
    
###################################Tensorflow Session################################################# 
#Run the Tensorflow graph on the GPU (Note, if system does not have a GPU or if Tensorflow-gpu is not
    #installed, change this line to '/cpu:0')
with tf.device('/gpu:0'):
    
    #Transform the label indices into a one-hot array.
    Labels = tf.cast(tf.one_hot(tf.cast(Z,dtype=tf.int32),10),dtype=tf.float32)

    #Add an aditional axis to the input to feed into the convolutional layers The last axis of
        #this tensor corresponds to the number of input channels. Adding this last axis essitially
        #tells the convolutional layer that there is 1 channel rather than 0 channels.
        
    #Normalize the inputs (which consit of a number between 0 and 255) to lie between 0 and 1.
    UU = tf.expand_dims(U,-1)/255
    
    #Define the convolutional neural network
    
    #[None,28,28,1] --> [None,14,14,8]
    _,Pool1 = Convolve(UU,3,8)
    #[None,14,14,8] --> [None,7,7,32]
    _,Pool2 = Convolve(Pool1,5,32)
    #[None,7,7,32] --> [None,4,4,64]
    _,Pool3 = Convolve(Pool2,3,64)

    #Define the fully connected network
    #[None,4,4,64] --> [None,512]
    L1 = Dense(Pool3,512,Input=[4,4,64])
    #[None,512]-->[None,128]
    L2 = Dense(L1,128)
    #[None,128]-->[None,64]
    L3 = Dense(L2,64)
    #[None,64]-->[None,10]
    L4 = Dense(L3,10,act='linear')
    
    #[None,10] Outputs a one-hot tensor correspond to the labels (only called during test phase)
    Out = tf.nn.softmax(L4)
    
    #Binary cross entropy loss function for the last layer.
    _Loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Labels,logits=L4)
    #Averges the loss 
    Loss = tf.reduce_sum(_Loss)/batchSize
    
    optimizer = tf.train.AdamOptimizer(.001).minimize(Loss)
        
###################################Initialize Session###################################        
#Configures the session to allow soft placement. (Operations that cannot be run on GPU will be
    #automatically run on CPU)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
#Initialize variables
sess.run(tf.global_variables_initializer())
#Initialize the datasets and feed the numpy arrays into tensorflow graph.
sess.run(iterator.initializer,feed_dict={_U:x_train,_Z:y_train})


def ErrorCalc(Out,Labels):
    """Averages the absolute difference error between the model's output and the labeled input. 
    Input:
        Out - [None,10] The output of the model is a tensor where axis 1 gives 
            the pseudo-probability of a given label being given to that image
        Labels - [None,10] A one-hot tensor representing the label corresponding with the image
    Outputs:
        The average error (average taken along axis 0)
    """
    return np.sum(abs(Out-Labels))/Out.shape[0]
###################################Run Session###################################
Losses = []

#Run the training process over the specified number of epochs.
for i in range(epochs):
    _,loss,o,l = sess.run([optimizer,Loss,Out,Labels])
    Losses.append(loss)
    print(i,loss)

#Print the error across the training set.
print("Train fit",ErrorCalc(o,l))
###################################Evaluate Results###################################
#Re-initialize the datasets. with the test data
sess.run(iterator.initializer,feed_dict={_U:x_test,_Z:y_test})

#Run the the test model with the inputs, outputs, labels, (and some intermediate tensors
    #for analysis.)
A = sess.run([UU[:,:,:,0],L1,L2,L3,Out,Labels])

#Run the error calculation for the test dataset.
error = ErrorCalc(A[-2],A[-1])
print("TestFit",error)


#Plot the results and the predictions for those results.  Run trough each sample in the batch
    #and prompt the user to press a key before iterating through to the next sample.

import matplotlib.pyplot as plt
for i in range(len(A[0])):
    
    plt.imshow(A[0][i])
    plt.show()
    print(np.argmax(A[-2][i]))
    a = input("Press any key to continue\nPress 'q' to quit")
    #Breaks out of the loop.
    if a == 'q':
        break
