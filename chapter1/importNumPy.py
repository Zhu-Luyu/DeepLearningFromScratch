import numpy as np

# NumPy的基本用法
x = np.array([1.0, 2.0, 3.0])
# print("x: " + x)
print(x)
y = np.array([2.0, 3.0, 4.0])
# element-wise product 乘法
print(x * y)
print(x / y)
# 如果x、y中元素个数不同，将会报错
# 但是NumPy有广播功能，某些条件下能够使用
z = x / 2.0
print(z)

# NumPy的N维数组/矩阵
a = np.array([[3, 4], [7, 9]])
print(a)
# 查看形状
print(a.shape)
# 查看元素的数据类型
print(a.dtype)
b = np.array([[-1, 5], [10, 0]])
print(a + b)
print(a / b)
# 广播
c = np.array([7,7])
print(a - c)
# 访问元素
print(a[0])
print(a[1][1])
for row in a:
    print(row)
# numPy还可以使用数组访问各个元素
A = a.flatten()
print(A)
print(A[np.array([1, 3])])
# 运用标记法，可以获取满足一定条件的元素
print(a > 5)
print(a[a > 5])
e = np.array([[1,9], [-2, 10]])
print(e[e > 5])
