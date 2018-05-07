#This is the architecture of 7-layer 2D CNN applied in the paper
#version1, 2018-05-05
import tensorflow as tf

# convolution
def conv2d(name, l_input, w, b, k):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, k, k, 1], padding='VALID'), b), name=name)

# pooling
def max_pool(name, l_input, k1, k2):
    return tf.nn.max_pool(l_input, ksize=[1, k1, k1, 1], strides=[1, k2, k2, 1], padding='VALID', name=name)

def CNN7(_X, _weights, _biases, _dropout):

    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'], 4)
    pool1 = max_pool('pool1', conv1, 3, 2)

    padded_pool1 = tf.pad(pool1, [0, 0], [2, 2], [2, 2], [0, 0])
    conv2 = conv2d('conv2', padded_pool1, _weights['wc2'], _biases['bc2'], 2)
    pool2 = max_pool('pool2', conv2, 3, 2)

    padded_pool2 = tf.pad(pool2, [0, 0], [1, 1], [1, 1], [0, 0])
    conv3 = conv2d('conv3', padded_pool2, _weights['wc3'], _biases['bc3'], 1)

    padded_conv3 = tf.pad(conv3, [0, 0], [1,1], [1, 1], [0, 0])
    conv4 = conv2d('conv4', padded_conv3, _weights['wc4'], _biases['bc4'], 2)
    pool4 = max_pool('pool4', conv4, 3, 2)

    conv5 = conv2d('conv5', pool4, _weights['wc5'], _biases['bc5'], 1)
    conv5 = tf.nn.dropout(conv5, _dropout)

    conv6 = conv2d('conv6', conv5, _weights['wc6'], _biases['bc6'], 1)
    conv6 = tf.nn.dropout(conv6, _dropout)

    padded_conv6 = tf.pad(conv6, [0, 0], [1, 1], [1, 1], [0, 0])
    conv7 = conv2d('conv7', padded_conv6, _weights['wc7'], _biases['bc7'], 2)
    conv7 = tf.nn.dropout(conv7, _dropout)

    # output layer
    out = tf.matmul(conv7, _weights['out']) + _biases['out']
    return out

#main function

# define parameter of the network
learning_rate = 1e-4
n_classes = 2
dropout = 0.5
n_input = 200 #? or 3

weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96])), #in_channel, 200*200*1
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wc5': tf.Variable(tf.random_normal([6, 6, 256, 192])),
    'wc6': tf.Variable(tf.random_normal([1, 1, 192, 96])),
    'wc7': tf.Variable(tf.random_normal([1, 1, 96, 3])),

    'out': tf.Variable(tf.random_normal([3, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([256])),
    'bc5': tf.Variable(tf.random_normal([192])),
    'bc6': tf.Variable(tf.random_normal([96])),
    'bc7': tf.Variable(tf.random_normal([3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder(tf.float32, [None, n_input]) # Inserts a placeholder for a tensor that will be always fed.
y = tf.placeholder(tf.float32, [None, n_classes]) #
keep_prob = tf.placeholder(tf.float32)

# prediction model
pred = CNN7(x, weights, biases, keep_prob)

# compute cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) # idk why

# test network
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# to be continued

#read and load data
#train with data,    input 4D tensor
#compute accuracy and loss
#This is the architecture of 7-layer 2D CNN applied in the paper
#version1, 2018-05-05
import tensorflow as tf

# convolution
def conv2d(name, l_input, w, b, k):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, k, k, 1], padding='VALID'), b), name=name)

# pooling
def max_pool(name, l_input, k1, k2):
    return tf.nn.max_pool(l_input, ksize=[1, k1, k1, 1], strides=[1, k2, k2, 1], padding='VALID', name=name)

def CNN7(_X, _weights, _biases, _dropout):

    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'], 4)
    pool1 = max_pool('pool1', conv1, 3, 2)

    padded_pool1 = tf.pad(pool1, [0, 0], [2, 2], [2, 2], [0, 0])
    conv2 = conv2d('conv2', padded_pool1, _weights['wc2'], _biases['bc2'], 2)
    pool2 = max_pool('pool2', conv2, 3, 2)

    padded_pool2 = tf.pad(pool2, [0, 0], [1, 1], [1, 1], [0, 0])
    conv3 = conv2d('conv3', padded_pool2, _weights['wc3'], _biases['bc3'], 1)

    padded_conv3 = tf.pad(conv3, [0, 0], [1,1], [1, 1], [0, 0])
    conv4 = conv2d('conv4', padded_conv3, _weights['wc4'], _biases['bc4'], 2)
    pool4 = max_pool('pool4', conv4, 3, 2)

    conv5 = conv2d('conv5', pool4, _weights['wc5'], _biases['bc5'], 1)
    conv5 = tf.nn.dropout(conv5, _dropout)

    conv6 = conv2d('conv6', conv5, _weights['wc6'], _biases['bc6'], 1)
    conv6 = tf.nn.dropout(conv6, _dropout)

    padded_conv6 = tf.pad(conv6, [0, 0], [1, 1], [1, 1], [0, 0])
    conv7 = conv2d('conv7', padded_conv6, _weights['wc7'], _biases['bc7'], 2)
    conv7 = tf.nn.dropout(conv7, _dropout)

    # output layer
    out = tf.matmul(conv7, _weights['out']) + _biases['out']
    return out

#main function

# define parameter of the network
learning_rate = 1e-4
n_classes = 2
dropout = 0.5
n_input = 200 #? or 3

weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96])), #in_channel, 200*200*1
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wc5': tf.Variable(tf.random_normal([6, 6, 256, 192])),
    'wc6': tf.Variable(tf.random_normal([1, 1, 192, 96])),
    'wc7': tf.Variable(tf.random_normal([1, 1, 96, 3])),

    'out': tf.Variable(tf.random_normal([3, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([256])),
    'bc5': tf.Variable(tf.random_normal([192])),
    'bc6': tf.Variable(tf.random_normal([96])),
    'bc7': tf.Variable(tf.random_normal([3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder(tf.float32, [None, n_input]) # Inserts a placeholder for a tensor that will be always fed.
y = tf.placeholder(tf.float32, [None, n_classes]) #
keep_prob = tf.placeholder(tf.float32)

# prediction model
pred = CNN7(x, weights, biases, keep_prob)

# compute cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) # idk why

# test network
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# to be continued

#read and load data
#train with data,    input 4D tensor
#compute accuracy and loss
#This is the architecture of 7-layer 2D CNN applied in the paper
#version1, 2018-05-05
import tensorflow as tf

# convolution
def conv2d(name, l_input, w, b, k):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, k, k, 1], padding='VALID'), b), name=name)

# pooling
def max_pool(name, l_input, k1, k2):
    return tf.nn.max_pool(l_input, ksize=[1, k1, k1, 1], strides=[1, k2, k2, 1], padding='VALID', name=name)

def CNN7(_X, _weights, _biases, _dropout):

    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'], 4)
    pool1 = max_pool('pool1', conv1, 3, 2)

    padded_pool1 = tf.pad(pool1, [0, 0], [2, 2], [2, 2], [0, 0])
    conv2 = conv2d('conv2', padded_pool1, _weights['wc2'], _biases['bc2'], 2)
    pool2 = max_pool('pool2', conv2, 3, 2)

    padded_pool2 = tf.pad(pool2, [0, 0], [1, 1], [1, 1], [0, 0])
    conv3 = conv2d('conv3', padded_pool2, _weights['wc3'], _biases['bc3'], 1)

    padded_conv3 = tf.pad(conv3, [0, 0], [1,1], [1, 1], [0, 0])
    conv4 = conv2d('conv4', padded_conv3, _weights['wc4'], _biases['bc4'], 2)
    pool4 = max_pool('pool4', conv4, 3, 2)

    conv5 = conv2d('conv5', pool4, _weights['wc5'], _biases['bc5'], 1)
    conv5 = tf.nn.dropout(conv5, _dropout)

    conv6 = conv2d('conv6', conv5, _weights['wc6'], _biases['bc6'], 1)
    conv6 = tf.nn.dropout(conv6, _dropout)

    padded_conv6 = tf.pad(conv6, [0, 0], [1, 1], [1, 1], [0, 0])
    conv7 = conv2d('conv7', padded_conv6, _weights['wc7'], _biases['bc7'], 2)
    conv7 = tf.nn.dropout(conv7, _dropout)

    # output layer
    out = tf.matmul(conv7, _weights['out']) + _biases['out']
    return out

#main function

# define parameter of the network
learning_rate = 1e-4
n_classes = 2
dropout = 0.5
n_input = 200 #? or 3

weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96])), #in_channel, 200*200*1
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wc5': tf.Variable(tf.random_normal([6, 6, 256, 192])),
    'wc6': tf.Variable(tf.random_normal([1, 1, 192, 96])),
    'wc7': tf.Variable(tf.random_normal([1, 1, 96, 3])),

    'out': tf.Variable(tf.random_normal([3, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([256])),
    'bc5': tf.Variable(tf.random_normal([192])),
    'bc6': tf.Variable(tf.random_normal([96])),
    'bc7': tf.Variable(tf.random_normal([3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder(tf.float32, [None, n_input]) # Inserts a placeholder for a tensor that will be always fed.
y = tf.placeholder(tf.float32, [None, n_classes]) #
keep_prob = tf.placeholder(tf.float32)

# prediction model
pred = CNN7(x, weights, biases, keep_prob)

# compute cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) # idk why

# test network
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# to be continued

#read and load data
#train with data,    input 4D tensor
#compute accuracy and loss
