import matplotlib.pyplot as plt
from matplotlib.image import imread

# 读入图像（设定合适的路径）
img = imread('../dataset/night_mountain.jpg')
plt.imshow(img)
plt.savefig('saveimg')