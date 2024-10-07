# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
# save_file = dataset_dir + "/mnist.pkl"
save_file = dataset_dir + "/MNIST/raw/train-images-idx3-ubyte.gz"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
    
def download_mnist():
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data
    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    # if not os.path.exists(save_file):
        # 如果本地文件不存在，则尝试使用 torch 来下载数据集
    # print("Local MNIST dataset not found, using torchvision to download...")
    transform = transforms.Compose([transforms.ToTensor()])
        
    train_set = torchvision.datasets.MNIST(root=dataset_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=dataset_dir, train=False, download=True, transform=transform)
        
    x_train = train_set.data.numpy()
    t_train = train_set.targets.numpy()
    x_test = test_set.data.numpy()
    t_test = test_set.targets.numpy()
        
    if flatten:
        x_train = x_train.reshape(train_num, -1)
        x_test = x_test.reshape(test_num, -1)
            
    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
            
        
    # 加载本地数据集（如果存在）
    # with open(save_file, 'rb') as f:
        # dataset = pickle.load(f)
    
    # if normalize:
    #     for key in ('train_img', 'test_img'):
    #         dataset[key] = dataset[key].astype(np.float32)
    #         dataset[key] /= 255.0
            
    if one_hot_label:
        t_train = _change_one_hot_label(t_train)
        t_test = _change_one_hot_label(t_test)
    
    # if not flatten:
    #      for key in ('train_img', 'test_img'):
    #         dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    # return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 
    return (x_train, t_train), (x_test, t_test)



if __name__ == '__main__':
    init_mnist()
