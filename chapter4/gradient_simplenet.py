import sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from chapter3.softmax import softmax
from loss_function import cross_entropy_error
# from numerical_differentiation import numerical_gradient
from deep_learning_from_scratch.common.gradient import numerical_gradient # 多维实现版

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 用高斯分布初始化
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        return loss
    
if __name__ == "__main__":
    net = simpleNet()
    print(net.W)
    
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))
    t = np.array([0,0,1])
    print(net.loss(x,t))
    
    # 求梯度
    # def f(_):
    #     return net.loss(x, t)
    f = lambda _: net.loss(x, t)
    print(numerical_gradient(f, net.W))