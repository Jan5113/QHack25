# -*- coding: utf-8 -*-
"""
Created on Sun May 11 02:09:08 2025

@author: Somesh
"""
import cvxpy as cp
import dynamiqs as dq
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax

from wigner_from_state_w_noise import wigner_from_state

# a = 0.5
# fock_state = dq.fock(5, 1)
# coherent_state = dq.coherent(3, 0.5)
# cat_state = dq.coherent(3, a) + dq.coherent(3, -a)
# cat_state = cat_state/dq.norm(cat_state)

n = 10
nx, ny = (n, n)
xmax = 2
x = jnp.linspace(-xmax, xmax, nx)
y = jnp.linspace(-xmax, xmax, ny)
xv, yv = jnp.meshgrid(x, y)

for k in range(2, 5):
    for l in range(5):
        state = dq.coherent(k, 0.5 + l/2) + dq.coherent(k, -0.5 - l/2)
        state = state/dq.norm(state)
        fidelity = []
        print(k, l)
        for sigma in jnp.linspace(0, 1, 11):
            rho_out, fidelity_step =\
                wigner_from_state(state, xv, yv, sigma)
            fidelity.append(fidelity_step)
        fig, ax = plt.subplots()
        ax.plot(jnp.linspace(0, 1, 11), fidelity)
        ax.set_xlabel('Noise [\sigma]')
        ax.set_ylabel('Fidelity')
        ax.set_title(f'Noise-Fidelity Plot for Cat State ({k}, {0.5+l/2})')
        plt.savefig("C:/Users/Somesh/OneDrive/Documents/ETH/10._Semester/50_Fun/QHC/QHack25/Images/Task_1_C_Cat/Cat_"+str(k)+"_"+str(l)+".png", dpi = 200)
        plt.show()

# fidelity_fock = []
# fidelity_coherent = []
# fidelity_cat = []

# # dq.plot.wigner(coherent_state)
# # dq.plot.wigner(cat_state)
# for sigma in jnp.linspace(0, 1, 10):
#     print(sigma)
#     rho_out_fock, fidelity_fock_step = wigner_from_state(fock_state, xv, yv, sigma)
#     rho_out_coherent, fidelity_coherent_step = wigner_from_state(coherent_state, xv, yv, sigma)
#     # dq.plot.wigner(rho_out_coherent)
#     rho_out_cat, fidelity_cat_step = wigner_from_state(cat_state, xv, yv, sigma)
#     # dq.plot.wigner(rho_out_cat)
#     fidelity_fock.append(fidelity_fock_step)
#     fidelity_coherent.append(fidelity_coherent_step)
#     fidelity_cat.append(fidelity_cat_step)
    
# fig, ax = plt.subplots()
# ax.plot(jnp.linspace(0, 1, 10), fidelity_fock, c = 'green')
# ax.plot(jnp.linspace(0, 1, 10), fidelity_coherent, c = 'blue')
# ax.plot(jnp.linspace(0, 1, 10), fidelity_cat, c = 'red')