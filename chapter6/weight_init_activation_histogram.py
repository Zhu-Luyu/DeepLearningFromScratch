# 权重初始化+sigmoid带来的梯度消失问题（Xavier初始化可解决）
# Xavier初始值是以激活函数是线性函数为前提而推导出来的
# 当激活函数使用ReLU时，一般推荐使用ReLU专用的初始值，He初始值
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    x = np.random.randn(1000, 100) # 1000个数据，每个数据100个特征
    node_num = 100 # 各隐藏层的节点（神经元）数
    hidden_layer_size = 5 # 隐藏层有5层
    activations = {} # 激活值的结果保存在这里
    
    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i - 1]
            
        # w = np.random.randn(node_num, node_num) * 1 # 标准正态分布（均值为 0，标准差为 1），68%的数据会落在 [-1, 1] 之间
        # w = np.random.randn(node_num, node_num) * 0.01 # 会把标准差从1变成0.01
        w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
        z = np.dot(x, w)
        # 因为权重是用标准差为1的高斯分布随机初始化的
        # 这导致输入 z 的值分布范围很广，从而让激活值集中在 0 和 1 附近。
        a = sigmoid(z)
        activations[i] = a
        
    for i, a in activations.items():
        plt.subplot(1, len(activations), i+1)
        plt.title(f"{i+1}-layer")
        plt.hist(a.flatten(), bins=30, range=(0,1))
        # a.flatten()这么做的意义主要是为了 方便绘制激活值的分布。
        # 虽然它打乱了各个样本之间的结构关系，
        # 但这里的目的是分析这些激活值在总体上的分布，而不是关注每个样本的激活值细节。
    plt.show()