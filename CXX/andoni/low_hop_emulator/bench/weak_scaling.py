import matplotlib.pyplot as plt
import numpy as np

f, ax = plt.subplots()

nodes             = [2,    8,    32,   64]
bvec              = [10.26,10.15,10.08,16.81]
multi             = [2.59, 2.21, 2.07, 2.60]

ax.plot(nodes, bvec,  label='bvec')
ax.plot(nodes, multi, label='multi')
ax.set_xlabel('nodes (64 ppn)')
ax.set_ylabel('runtime (s)')
ax.set_title("Edge Weak Scaling -sp 0.005 -b 16")
ax.legend()

plt.show()
