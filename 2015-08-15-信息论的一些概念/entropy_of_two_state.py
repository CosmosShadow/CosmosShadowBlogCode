import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.001, 0.999, 999)
y = -x*np.log(x) - (1-x)*np.log(1-x)
plt.figure(figsize=(8, 4))
plt.plot(x, y, label="$y = -x*np.log(x) - (1-x)*np.log(1-x)$", color="red")
# plt.ylim(-1.2, 1.2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Entropy of Two State")
plt.legend()
plt.show()