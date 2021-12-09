import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative

def f(x): return 20e9+1.8e9*np.sqrt(np.abs(x))
derivative(f,1,dx=0.01)
xs=np.linspace(0,12,100)
plt.plot(xs,f(xs))
# plt.plot(xs,derivative(f,xs, dx=0.01))
ax = plt.gca()
# plt.plot(xs, 20e9+1.82e9*np.sqrt( abs( np.cos( xs ) ) ))

# ax.set_ylim(-5,5)
plt.savefig('temp.png')
plt.show()
