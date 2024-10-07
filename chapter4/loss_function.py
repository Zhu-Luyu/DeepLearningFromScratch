import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from deep_learning_from_scratch.dataset.mnist import load_mnist

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error_(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def cross_entropy_error(y, t, one_hot_label=True):
    delta = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    if one_hot_label:
        return -np.sum(t * np.log(y + delta)) / batch_size
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    # 当监督数据t是标签形式
    # t：目标标签（真实值），是一维整数数组，表示类别标签，形状为 (batch_size, )。
    # y：模型的输出（预测值），通常是经过 softmax 的输出概率，二维数组形状为 (batch_size, num_classes)。

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    print(f"x_train.shape:{x_train.shape}")
    print(f"t_train.shape:{t_train.shape}")
    print(f"x_train.size:{x_train.size}")
    print(f"t_train.size:{t_train.size}")
    print(f"t_train:{t_train}")
    
    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print(f"batch_mask:{batch_mask}")