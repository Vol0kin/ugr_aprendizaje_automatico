import numpy as np
import matplotlib.pyplot as plt

class1 = np.array([[1, 1], [2, 2], [2, 0]])
class2 = np.array([[0, 0], [1, 0], [0, 1]])
middle = np.array([[0.5, 1], [1.5, 0]])

plt.scatter(class1[:, 0], class1[:, 1], c='blue', label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], c='red', label='Class 2')
#plt.scatter(middle[:, 0], middle[:, 1], c='black', label='Middle points')
plt.plot([-1, 2.5], [2.5, -1.0], 'k-', label='Hyperplane')
plt.plot([-1, 2], [2, -1.0], 'g--', label='Margin')
plt.plot([-1, 3], [3, -1.0], 'g--' )
plt.ylim(-1, 3)
plt.xlim(-1, 3)
plt.legend()

plt.show()
