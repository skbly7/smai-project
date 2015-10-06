import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import matplotlib
matplotlib.rc("axes", edgecolor="w")
matplotlib.rc("xtick", color="w")
matplotlib.rc("ytick", color="w")
#X = [100, 500, 1000, 5000, 10000, 50000, 100000]
#T = [9.197, 9.232, 9.363, 9.714, 11.040, 20.059, 40.98]
X = [41, 21, 11]
A = [98.19, 86.73, 84.37]
T = [11.040, 8.53, 7.99]

plt.plot(X,A, color="w", linewidth="2")
savefig("ffr_acc.png", transparent = True)
plt.clf()
plt.plot(X,T, color="w", linewidth="2")
savefig("time.png", transparent = True)
