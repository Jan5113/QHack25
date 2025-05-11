# -*- coding: utf-8 -*-
"""
Created on Sat May 10 18:07:32 2025

@author: Somesh
"""

import cvxpy as cp
import dynamiqs as dq
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax


def add_gaussian_noise(wigner, sigma):
    noise = sigma*jax.random.normal(jax.random.key(0), wigner.shape)
    return jnp.add(wigner, noise)

def wigner_from_state(rho_in,
           xv, yv,
           sigma):
    
    #Define Wigner Function
    def w(state, xv, xy):
        return jnp.pi/2*dq.wigner(state, xmax = xv.max(), ymax = yv.max(), npixels = len(xv))[2]
    
    #============================================================
    #Define Preliminary Data
    dim = rho_in.dims[0]
    if dim<5:
        fac = 10
    else:
        fac = 5
    
    #Check if dim has correct properties:
    if type(dim)!=int:
        raise('Dimension is not integer')
    
    #Compute measurement data
    w_k = add_gaussian_noise(w(rho_in, xv, yv), sigma).flatten()
    
    #Define Observables
    alpha = (xv.T + yv.T*1j).flatten()
    E_list = (dq.displace(dim*fac, alpha)@dq.parity(dim*fac)@dq.dag(dq.displace(dim*fac, alpha))).data[:, :dim, :dim]
    
    #===============================================================
    #Inititalize Optimization
    
    # Define variable
    rho = cp.Variable((dim, dim), hermitian=True)
    
    # Objective
    exprs = cp.hstack([cp.real(cp.trace(E@rho)) - k for E, k in zip(E_list, w_k)])
    
    # objective = cp.Minimize(cp.sum_squares(exprs))
    objective = cp.Minimize(cp.sum(cp.huber(exprs)))
    
    # print(objective)
    
    # Constraints
    constraints = [rho >> 0, cp.trace(rho) == 1]
    
    # Solve problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK)
    
    # Output estimated density matrix
    rho_out = rho.value
    #Compute Fidelity
    fidelity = dq.fidelity(rho_in, rho_out)
    
    return rho_out, fidelity

# a = 0.5
# fock_state = dq.fock(5, 1)
# coherent_state = dq.coherent(3, 0.5)
# cat_state = dq.coherent(3, a) + dq.coherent(3, -a)
# cat_state = cat_state/dq.norm(cat_state)

# n = 10
# nx, ny = (n, n)
# xmax = 2
# x = jnp.linspace(-xmax, xmax, nx)
# y = jnp.linspace(-xmax, xmax, ny)
# xv, yv = jnp.meshgrid(x, y)

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
    
    

    
        