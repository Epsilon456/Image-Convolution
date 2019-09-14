import tensorflow as tf
import numpy as np

#Import data
(x_train, y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

##########################Data Sets#################################################
batchSize = 64
epochs = 10000
#Define placeholders
_U = tf.placeholder(tf.float32,shape=[None,28,28])
_Z = tf.placeholder(tf.int32,shape=[None])

#Define datasets
dU = tf.data.Dataset.from_tensor_slices(_U)
dZ = tf.data.Dataset.from_tensor_slices(_Z)
dataset = tf.data.Dataset.zip((dU,dZ))

#Shuffle and repeat if training
#trainQ = int(input("Train?:\n0)Train\n1)Test"))
trainQ = 0
if trainQ == 0:
    Dataset = dataset.batch(batchSize).shuffle(300).prefetch(128).repeat()
else:
    Dataset = dataset.batch(batchSize)
    
####################################Set iterator###################################
iterator = Dataset.make_initializable_iterator()
U,Z = iterator.get_next()


def He(fanin):
    return 1/fanin

def Convolve(inputTensor,width,filters):
    inputSize = int(inputTensor.shape[-1])
    variance = width**2*inputSize
    WC = tf.Variable(tf.truncated_normal([width,width,inputSize,filters],stddev=He(variance),dtype=tf.float32))
    BC = tf.Variable(tf.truncated_normal((filters,),stddev = He(filters),dtype=tf.float32))
    
    CV = tf.nn.conv2d(inputTensor,WC,[1,1,1,1],"SAME")+BC
    Pool = tf.nn.leaky_relu(tf.nn.pool(CV,[2,2],"MAX","SAME",strides=[2,2]),.01)
    
    return CV,Pool

def Dense(inputTensor,outShape,act='leaky_relu'):
    if type(outShape) == list:
        v = outShape
        variance = v[0]*v[1]*v[2]
        W = tf.Variable(tf.truncated_normal(outShape,stddev=He(variance),dtype=tf.float32))
        B = tf.Variable(tf.truncated_normal((v[-1],),dtype=tf.float32,stddev=He(v[-1])))
        einsum = 'ijkl,jklm->im'
    else:
        inShape = int(inputTensor.shape[-1])
        W = tf.Variable(tf.truncated_normal([inShape,outShape],stddev=He(inShape),dtype=tf.float32))
        B = tf.Variable(tf.truncated_normal((outShape,),dtype=tf.float32,stddev=He(outShape)))
        einsum = 'ij,jk->ik'
    
    if act == 'leaky_relu':
        return tf.nn.leaky_relu(tf.einsum(einsum,inputTensor,W)+B,.01)
    elif act == 'linear':
        return tf.einsum(einsum,inputTensor,W)+B
    
###################################Tensorflow Session################################################# 
with tf.device('/gpu:1'):
    Labels = tf.cast(tf.one_hot(tf.cast(Z,dtype=tf.int32),10),dtype=tf.float32)

    UU = tf.expand_dims(U,-1)/255
    
    CV1,Pool1 = Convolve(UU,3,8)
    CV2,Pool2 = Convolve(Pool1,5,32)
    CV3,Pool3 = Convolve(Pool2,3,64)

    L1 = Dense(Pool3,[4,4,64,512])
    L2 = Dense(L1,128)
    L3 = Dense(L2,64)
    L4 = Dense(L3,10,act='linear')
    
    
    Out = tf.nn.softmax(L4)
    
#    _Loss = tf.abs(Out-Labels)
    _Loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Labels,logits=L4)
    Loss = tf.reduce_sum(_Loss)/batchSize
    
    optimizer = tf.train.AdamOptimizer(.001).minimize(Loss)
    
   
    """Code here"""    
    
###################################Initialize Session###################################        
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
if trainQ == 0:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer,feed_dict={_U:x_train,_Z:y_train})
else:
    sess.run(iterator.initializer,feed_dict={_U:x_test,_Z:y_test})


def ErrorCalc(Out,Labels):
    return np.sum(abs(Out-Labels))/Out.shape[0]
###################################Run Session###################################
Losses = []
for i in range(epochs):
    
    _,loss,o,l = sess.run([optimizer,Loss,Out,Labels])
    Losses.append(loss)
    print(i,loss)

print("Train fit",ErrorCalc(o,l))
###################################Evaluate Results###################################
sess.run(iterator.initializer,feed_dict={_U:x_test,_Z:y_test})

A = sess.run([UU[:,:,:,0],L1,L2,L3,Out,Labels])

error = ErrorCalc(A[-2],A[-1])
print("TestFit",error)

import matplotlib.pyplot as plt
for i in range(len(A[0])):
    
    plt.imshow(A[0][i])
    plt.show()
    print(np.argmax(A[-2][i]))
    a = input("Press any key to continue\nPress 'q' to quit")
    if a == 'q':
        break

    
B = sess.run([CV1,Pool1,CV2,Pool2,CV3,Pool3])

for i in range(6):
    for j in range(8):
        b = B[i][0,:,:,j]
        plt.imshow(b)
        plt.show()

b = B[-1]
c = B[0]
plt.imshow(b[i,:,:,0])
plt.show()
plt.imshow(c[i,:,:,0])
plt.show()
i +=1