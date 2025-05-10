# -*- coding: utf-8 -*-
"""
Created on Sat May 10 14:40:15 2025

@author: Somesh
"""
import cvxpy as cp
import dynamiqs as dq
import jax.numpy as jnp
nx, ny = (6, 6)
x_del, y_del = (1, 1)
dim = 3

x = jnp.linspace(-x_del, x_del, nx)
y = jnp.linspace(-y_del, y_del, ny)
xv, yv = jnp.meshgrid(x, y)

def w(state, nx, x_del, y_del):
    return 1/2*(1+jnp.pi/2*dq.wigner(
        state,
        xmax = x_del,
        ymax = y_del,
        npixels = nx)[2])

coherent = dq.coherent(dim, 0.5)

w_k = w(coherent, nx, x_del, y_del).flatten()

alpha = (xv + yv*1j).flatten()
E_list = [dq.eye(dim) + dq.displace(dim, alpha_k)@dq.parity(dim)@dq.dag(dq.displace(dim, alpha_k)) for alpha_k in alpha]

# Define variables
rho = cp.Variable((dim, dim), hermitian=True)

# Objective
residuals = [cp.real(cp.trace(E @ rho)) - p for E, p in zip(E_list, w_k)]
objective = cp.Minimize(cp.sum_squares(cp.hstack(residuals)))

# Constraints
constraints = [rho >> 0, cp.trace(rho) == 1]

# Solve problem
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS)  # or use MOSEK if available

# Output estimated density matrix
rho_est = rho.value
print("Estimated density matrix ρ:")
print(rho_est)