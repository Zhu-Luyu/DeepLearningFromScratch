import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from deep_learning_from_scratch.dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pylab as plt

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    
    train_loss_list = []
    
    train_acc_list = []
    test_acc_list = []
    
    
    # 超参数
    iter_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learn_rate = 0.1
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    
    iter_per_epoch = max(train_size / batch_size, 1)
    
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
        
        # 计算每个epoch的ACC
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test ac | " + str(train_acc) + "," + str(test_acc))
        
    # 绘制图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制训练损失的变化
    ax1.plot(np.arange(len(train_loss_list)), train_loss_list, label='train loss', color='blue')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Iterations')
    # ax1.grid(True)

    # 绘制训练和测试准确率的变化
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    ax2.plot(x, train_acc_list, label='train acc', marker=markers['train'], color='green')
    ax2.plot(x, test_acc_list, label='test acc', marker=markers['test'], linestyle='--', color='red')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy Over Epochs')
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc='lower right')
    # ax2.grid(True)

    # 调整布局以避免重叠
    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.close()