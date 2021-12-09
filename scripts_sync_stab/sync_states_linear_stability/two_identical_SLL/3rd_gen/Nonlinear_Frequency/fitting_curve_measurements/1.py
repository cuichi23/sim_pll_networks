import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import numpy as np

def func(x, a, b):
    return a +b *np.sqrt(np.array(x) )

def func2(t):
    return 2.081500299e10 +2.19747032e9*np.sqrt(t )
t=np.linspace(1,5,20)
Vfine = []; wout = [];

with open('1.txt','r') as csvfile:
	plots = csv.reader(csvfile, delimiter='\t')
	for row in plots:
		Vfine.append(float(row[0]))
		# y1.append(float(row[2]))
		wout.append(float(row[1]))
# file.close()
xdata=Vfine;
ydata=wout;
popt, pcov = curve_fit(func, xdata, ydata, p0=[2.0e10, 1.8e9], bounds=(0, np.inf))
print(popt)# popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
popt
plt.plot(t, func2(t), '-', color='blue', label='2.081500299e10 +2.19747032e9*Sqrt(Vtune)')
plt.plot(xdata, ydata, 'o', label='data')
plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f ' % tuple(popt))
plt.legend()
plt.show()
