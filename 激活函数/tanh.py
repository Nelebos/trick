import numpy as np
import matplotlib.pylab as plt

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

x = np.arange(-10, 10, 0.1)
p1 = plt.subplot(111)
y = tanh(x)

plt.title('tanh')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.ylim(-1.0, 1.0)
plt.show()
