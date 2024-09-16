import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 生成示例数据
f = open('/home/kitil/VO_exercise/src/VO/src/cloud.txt')
ls = f.readlines()
x = []
y = []
z = []
for i in ls:
    print(i)
    xyz = i.split(',')
    if float(xyz[2]) < 200:
        if abs(float(xyz[0])) < 200:
            if abs(float(xyz[1])) < 200:
                x.append(float(xyz[0]))
                y.append(float(xyz[1]))
                z.append(float(xyz[2]))

# 创建一个图形对象
fig = plt.figure()

# 添加一个3D子图
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图
ax.scatter(x, y, z, c='r', marker='.')

# 添加标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图形
plt.show()
