import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from deep_learning_from_scratch.dataset.mnist import load_mnist
import pickle
import numpy as np
from sigmoid import sigmoid
from softmax import softmax

def get_data():
    (x_train,t_train), (x_test,t_test) = load_mnist(flatten=True, normalize=True,one_hot_label=False)
    return x_test, t_test

def init_network():
    with open('sample_weight.pkl','rb') as f:
        network = pickle.load(f)
        
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    
    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
batch_size = 100
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch,axis=1) # 获取概率最高的元素的索引
    accuracy_cnt += sum(p==t[i:i+batch_size])
        
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
    