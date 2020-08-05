from matplotlib import pyplot as plt
import numpy as np
import math

def sigmoid_function(x):
    fx = []
    for num in x:
        fx.append(1 / (1 + math.exp(-num)))
    return fx

x = np.arange(-10, 10, 0.01)
fx = sigmoid_function(x)

plt.title('Sigmoid')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, fx)
plt.show()
