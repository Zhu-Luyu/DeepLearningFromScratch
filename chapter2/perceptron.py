# 感知机
# 重点内容：感知机的结构；与、或、与非门的参数设置

import numpy as np

# 先定义一个接收参数x1和x2的AND函数
def AND_(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
    
print(AND_(0,0),AND_(0,1),AND_(1,0),AND_(1,1))

# 使用NumPy库，并将theta等价为偏置
# 与门
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7    # 偏置这个术语，有“穿木屐”的效果
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
print(AND(0,0),AND(0,1),AND(1,0),AND(1,1))

# 与非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
print(NAND(0,0),NAND(0,1),NAND(1,0),NAND(1,1))

# 或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
print(OR(0,0),OR(0,1),OR(1,0),OR(1,1))

# 感知机的局限在于它只能表示由一条直线分割的空间
# 即将空间分割为线性空间
# 后面会提到，感知机通过叠加层能够进行非线性的表示

# 通过组合与门、与非门、或门实现异或门
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0,0),XOR(0,1),XOR(1,0),XOR(1,1))

# 从神经元的结构去看，这是多层（2层）感知机
# 单层感知机无法表示的东西，通过增加一层就可以解决
# 第0层的两个神经元接收输入信号，并将信号发送至第1层的神经元，
# 第1层的神经元将信号发送至第2层的神经元，第2层的神经元输出y