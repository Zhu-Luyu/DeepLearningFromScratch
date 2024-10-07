import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from deep_learning_from_scratch.dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pylab as plt

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    
    train_loss_list = []
    
    # 超参数
    iter_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learn_rate = 0.1
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    
    for i in range(iter_num):
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 计算梯度
        grad = network.numerical_gradient(x_batch, t_batch)
        
        # 更新参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learn_rate * grad[key]
            
        # 记录学习过程
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
    x = np.arange(0, iter_num, 1) 
    plt.xlabel("iter_num")
    plt.ylabel("loss")
    plt.plot(x,train_loss_list)
    plt.show()