import numpy as np
import matplotlib.pylab as plt

# 不好的实现示例
def numerical_diff_(f,x):
    h = 10e-50
    return (f(x+h) - f(x)) / h

def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return x[0]**2 + x[1]**2
    # 或者 return np.sum(x**2)
    
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值
    return grad
        
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x=init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad
    return x
    


if __name__ == "__main__":
    x = np.arange(0.0, 20.0, 0.1) 
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x,y)
    # plt.show()
    
    print(numerical_diff(function_1, 5)) # 0.1999999999990898
    print(numerical_diff(function_1, 10)) # 0.2999999999986347
    # 真的导数：0.2和0.3，上述计算结果和解析性求导误差非常小  
    
    # 求偏导
    x0,x1=3.0,4.0
    # 定义新函数
    def function_2_tmp1(x0):
        return x0*x0 + 4.0**2.0  
    print(numerical_diff(function_2_tmp1,x0)) # 6.00000000000378
    def function_2_tmp2(x1):
        return 3.0**2.0 + x1*x1
    print(numerical_diff(function_2_tmp2,x1)) # 7.999999999999119
    
    print(numerical_gradient(function_2, np.array([3.0,4.0]))) # [6. 8.]
    print(numerical_gradient(function_2, np.array([0.0,2.0]))) # [0. 4.]
    print(numerical_gradient(function_2, np.array([3.0,0.0]))) # [6. 0.]
    
    # 梯度法求最小值
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)) # [-6.11110793e-10  8.14814391e-10]
    
