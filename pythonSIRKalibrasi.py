# -*- coding: utf-8 -*-
"""
Created on Wed May 20 02:17:45 2020

@author: ASUS
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import csv
import pandas as pd

dataset = pd.read_csv('save_progress.csv')
# Total population, N.
N = 763
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 1.67, 1./2
# A grid of time points (in days)
t = np.linspace(0, 14, 14)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

#save data
save_data = open('save_progress2_NPOP1158_NINF5_R06.csv', 'w')
save_data.write('t,S,I,R\n')
for i in range(len(t)):
	str_save = '%s,%s,%s,%s\n'%(round(t[i]), round(S[i]), round(I[i]), round(R[i]))
	save_data.write(str_save)
save_data.close()

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time steps')
ax.set_ylabel('Number (1000s)')
ax.set_ylim(0,2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
