import matplotlib.pyplot as plt
import numpy as np
f1 = 26
f=42
lw = 9
mw = 9

'''
b g r c y m k

c y g
r m b
'''

##FOR X-Y SCATTER PLOT

######
acc = np.array([95.299,92.308,82.478,89.744,84.188,79.060,80.769,90.598,82.478,84.188,88.462,79.487,82.478,88.034,82.906,80.342,76.923,80.342,81.624,80.342,78.632])
x = np.arange(21)
sd = np.array([1.480,1.282,0.740,3.392,1.958,0.740,4.622,3.227,2.669,2.669,0.000,2.220,2.669,1.959,2.669,4.121,2.220,1.480,3.701,2.669,3.226])
std = np.array([(np.zeros(len(sd))),(sd)])
plt.errorbar(x,acc,yerr = std, fmt='-o', ecolor = 'c', color = 'r', mfc = 'k', capthick =3, linewidth = lw, mew=mw, capsize = 14, elinewidth=9)
plt.xlabel('Random Integer N', fontsize = f)
plt.axis([0, 20, 0, 100])



plt.ylabel('Performance Accuracy', fontsize = f)
plt.xticks(fontsize=f1)
plt.yticks(fontsize=f1)
plt.grid(True)
plt.show()
'''

##FOR BAR GRAPH PLOT
acc = np.array([98.718,26.068])
ind = np.arange(len(acc))
sd = np.array([1.282, 4.854])     
peakval = ['98.718','26.068'] 

stds    = [(0,0), sd] # Standard deviation Data
width = 0.3
colours = ['red','blue']

plt.figure()
plt.bar(ind, acc, width, color=colours, align='center', yerr=stds, ecolor='k',error_kw=dict(lw=4, capsize=14, capthick=3))
plt.ylabel('Performance Accuracy', fontsize = f)
plt.yticks(fontsize=f1)
plt.xticks(ind,("First x", "Full with First x (2x)"), fontsize = f)
def autolabel(bars,peakval):
    for ii,bar in enumerate(bars):
        height = bars[ii]
        plt.text(ind[ii], height-8, '%s'% (peakval[ii]), ha='center', va='bottom', fontsize = f)
autolabel(acc,peakval) 
plt.show()
'''