import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import matplotlib
matplotlib.rc("axes", edgecolor="w")
matplotlib.rc("xtick", color="w")
matplotlib.rc("ytick", color="w")
X = [41, 21, 11]
A = [98.19, 90.93, 89.36]
T = [52, 13.8, 14]

plt.plot(X,A, color="w", linewidth="2")
savefig("acc.png", transparent = True)
plt.clf()
plt.plot(X,T, color="w", linewidth="2")
savefig("time.png", transparent = True)