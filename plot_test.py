import matplotlib.pyplot as plt
import numpy as np

#### test 1 ####
# fr = np.array([10, 20, 30, 40])
# p_acc = np.array([82.906, 91.026, 93.163, 95.727])
# plt.plot(fr, p_acc, linestyle='-', marker='o', mfc = 'k', color='b')
# plt.xlabel('Output Frame')
# plt.ylabel('Performance Accuracy')
# plt.grid(True)
# plt.axis([0, 50, 70, 100])
# plt.show()

#### test 2 ####
# x = np.arange(2)
# p_acc = np.array([94.445, 48.718])
# rs, sp = plt.bar(x, p_acc)
# rs.set_facecolor('r')
# sp.set_facecolor('g')
# plt.xticks(x, ('Random Split', 'Split By-Person'))
# plt.xlabel('Method of Splitting Train-Test data')
# plt.ylabel('Performance Accuracy')
# plt.show()

#### test 3 ####
# x = np.arange(4)
# p_acc = np.array([99.145, 22.650, 45.727, 98.291])
# za, zp, ff, lf = plt.bar(x, p_acc)
# za.set_facecolor('r')
# zp.set_facecolor('g')
# ff.set_facecolor('b')
# lf.set_facecolor('c')
# plt.xticks(x, ('Zero Append', 'Zero Prepend', 'First Frame Prepend', 'Last Frame Append'))
# plt.ylabel('Performance Accuracy')
# plt.show()

#### test 4 ####
x = np.arange(21)
p_acc = np.array([98.291,96.154,95.727,93.163,94.017,89.316,78.632,74.786,74.359,64.103,65.812,59.829,61.538,55.128,48.718,49.573,45.727,47.009,43.590,39.316,42.735])
plt.plot(x, p_acc, linestyle='-', marker='o', mfc = 'k', color='b')
plt.xlabel('Number of Frames')
plt.ylabel('Performance Accuracy')
plt.grid(True)
plt.axis([0, 25, 0, 100])
plt.show()