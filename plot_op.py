'''
================    ===============================
character           description
================    ===============================
   -                solid line style
   --               dashed line style
   -.               dash-dot line style
   :                dotted line style
   .                point marker
   ,                pixel marker
   o                circle marker
   v                triangle_down marker
   ^                triangle_up marker
   <                triangle_left marker
   >                triangle_right marker
   1                tri_down marker
   2                tri_up marker
   3                tri_left marker
   4                tri_right marker
   s                square marker
   p                pentagon marker
   *                star marker
   h                hexagon1 marker
   H                hexagon2 marker
   +                plus marker
   x                x marker
   D                diamond marker
   d                thin_diamond marker
   |                vline marker
   _                hline marker
================    ===============================
'''

import matplotlib.pyplot as plt
import numpy as np
plt.plot(range(10), linestyle='-', marker='o', mfc = 'k', color='b')
#plt.plot(range(10), '-bo')

# fig = plt.figure()
# ax = fig.add_subplot(111)

#fr = np.array([10, 20, 30, 40])
#p_acc = np.array([82.906, 91.026, 93.163, 95.727])
#e = np.array([3.917, 1.282, 1.480, 2.669])

#plt.errorbar(fr, p_acc, yerr=e, fmt='-o', lolims=False)
#plt.plot(fr, p_acc, '-bo')

# for xy in zip(fr, p_acc):                                       # <--
#     ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

#plt.xlabel('Output Frame')
#plt.ylabel('Performance Accuracy')
#plt.grid(True)
#plt.axis([0, 50, 80, 100])
plt.show()

'''
y=[2.56422, 3.77284,3.52623,3.51468,3.02199]
z=[0.15, 0.3, 0.45, 0.6, 0.75]
n=[58,651,393,203,123]

fig, ax = plt.subplots()

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i],y[i]))
'''