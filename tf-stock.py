# Import
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 


# Import data
data = pd.read_csv('data/data_stocks.csv')
print(data.head())
# Drop date colmun in columns（axis=1), defautl in rows 
data = data.drop(['DATE'], 1)

plt.plot(data['SP500'])
plt.show()

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]
# Make data a numpy array
data = data.values

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))

test_start = train_end + 1
test_end = n

data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# Scale data
# Most common activation functions of the network’s neurons such as tanh or sigmoid are 
# defined on the [-1, 1] or [0, 1] interval respectively
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
# Build X and y
X_train = data_train[:, 1:]   # 1 to end cols
y_train = data_train[:, 0]    # first col
X_test = data_test[:, 1:]
y_test = data_test[:, 0]


# Number of stocks in training data
n_stocks = X_train.shape[1]

# Neurons
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128

# Session
net = tf.InteractiveSession()

# Placeholder
# The None argument indicates that at this point we do not yet know the number of observations 
# that flow through the neural net graph in each batch
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks]) # a 2-dimensional matrix
Y = tf.placeholder(dtype=tf.float32, shape=[None])

