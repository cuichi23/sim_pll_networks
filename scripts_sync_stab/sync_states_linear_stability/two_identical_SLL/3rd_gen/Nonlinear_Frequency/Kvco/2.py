import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

>>>

def func(x, a, b, c):
    return a+b*sqrt(x+c)
