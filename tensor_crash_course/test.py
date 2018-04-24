import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(1, 5, 100);

y1 = 2 * x

y2 = 5 * x + 30


plt.figure()
plt.plot(x, y1)
plt.show()


plt.figure()
plt.plot(x, y2)
plt.show()
