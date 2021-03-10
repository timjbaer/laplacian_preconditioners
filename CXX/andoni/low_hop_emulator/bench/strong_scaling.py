import matplotlib.pyplot as plt
import numpy as np

f, ax = plt.subplots()

nodes             = [1,    2,    4,    8]
bvec              = [10.26,10.15,10.08,16.81]
multi             = [2.59, 2.21, 2.07, 2.60]
bvec_conv         = [4.67, 4.45, 3.83, 7.70]
multi_conv        = [1.25, 1.05, 0.86, 1.22]
bvec_square       = [10.16,10.05,9.90, 16.6]
multi_square      = [2.59, 2.19, 2.06, 2.49]
bvec_conv_square  = [3.14, 2.85, 2.72, 6.48]
multi_conv_square = [1.50, 0.99, 0.93, 0.80]

ax.plot(nodes, bvec,              label='bvec'             )
ax.plot(nodes, multi,             label='multi'            )
ax.plot(nodes, bvec_conv,         label='bvec_conv'        )
ax.plot(nodes, multi_conv,        label='multi_conv'       )
ax.plot(nodes, bvec_square,       label='bvec_square'      )
ax.plot(nodes, multi_square,      label='multi_square'     )
ax.plot(nodes, bvec_conv_square,  label='bvec_conv_square' )
ax.plot(nodes, multi_conv_square, label='multi_conv_square')
ax.set_xlabel('nodes (64 ppn)')
ax.set_ylabel('runtime (s)')
ax.set_title("Strong Scaling -S 16 -E 10 -b 16")
ax.legend()

plt.show()
