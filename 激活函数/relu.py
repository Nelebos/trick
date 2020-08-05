import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0, x)  #输入的数据中选择较大的输出
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.title('Relu')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.show()

