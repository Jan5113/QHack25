# -*- coding: utf-8 -*-
"""
Created on Sat May 10 14:40:15 2025

@author: Somesh
"""
import cvxpy as cp
import dynamiqs as dq
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

data_path = "C:/Users/Somesh/OneDrive/Documents/ETH/10._Semester/50_Fun/QHC/QHack25/data/experimental/wigner_fock_zero.pickle"
data = pd.read_pickle(data_path)

dim = 3

x = data[0]
y = data[1]
xv, yv = jnp.meshgrid(x, y)

def w(state, nx):
    return 1/2*(1+jnp.pi/2*dq.wigner(
        state,
        npixels = nx)[2])

#===========================================================================
print('weights loading')
# w_k = w(coherent, nx, x_del, y_del).flatten()
w_k = data[2][::2, ::2].flatten()
print(len(w_k))

#Define Observables
alpha = (xv + yv*1j).flatten()
E_list = 1/2*(dq.eye(dim) + dq.displace(dim, alpha)@dq.parity(dim)@dq.dag(dq.displace(dim, alpha)))

# Define variables
rho = cp.Variable((dim, dim), hermitian=True)

print('Objectives Loading')
# Objective
residuals = [cp.real(cp.trace(E @ rho)) - p for E, p in zip(E_list, w_k)]
objective = cp.Minimize(cp.sum_squares(cp.hstack(residuals)))

# Constraints
constraints = [rho >> 0, cp.trace(rho) == 1]

print('Problem Solving')
# Solve problem
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK,
           verbose = True)  # or use MOSEK if available

# Output estimated density matrix
rho_est = rho.value
print("Estimated density matrix ρ:")
print(rho_est)

nx = len(data[2][::2, ::2])
w_est = w(rho_est, nx)
plt.imshow(w_est)
plt.imshow(data[2][::2, ::2])