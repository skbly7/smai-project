import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import matplotlib
matplotlib.rc("axes", edgecolor="w")
matplotlib.rc("xtick", color="w")
matplotlib.rc("ytick", color="w")
#X = [100, 500, 1000, 5000, 10000, 50000, 100000]
#T = [9.197, 9.232, 9.363, 9.714, 11.040, 20.059, 40.98]
X = [11, 21, 41]
A = [98.19, 90.93, 89.36]
T = [11.040, 9.363, 9.197]

plt.plot(X,T, color="w", linewidth="2")
savefig("acc.png", transparent = True)
plt.clf()
plt.plot(X,T, color="w", linewidth="2")
savefig("time.png", transparent = True)
