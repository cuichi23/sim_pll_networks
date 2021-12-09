import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import numpy as np

def func(x, a, b, c):
    return np.asarray(a) * np.exp(-(np.asarray(b) * np.asarray(x))**2) + np.asarray(c)

Vcoarse = []; Vfine = []; Kout = [];

with open('Sensitivity_fine.txt','r') as csvfile:
	plots = csv.reader(csvfile, delimiter='\t')
	for row in plots:
		Vcoarse.append(float(row[0]))
		Vfine.append(float(row[1]))
		# y1.append(float(row[2]))
		Kout.append(float(row[2]))
# file.close()
xdata=Vfine;
ydata=Kout;
popt, pcov = curve_fit(func, xdata, ydata)
# popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
plt.plot(xdata, ydata, 'b-', label='data')
plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
