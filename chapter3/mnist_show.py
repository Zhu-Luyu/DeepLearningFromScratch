import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deep_learning_from_scratch.dataset.mnist import load_mnist
from PIL import Image
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# 输出各个数据的形状
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    
img=x_train[0]
lable=t_train[0]

print(lable)

print(f'img.shape:{img.shape}')
img = img.reshape(28,28)
print(f'img.shape:{img.shape}')

img_show(img)