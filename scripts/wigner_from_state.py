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


def wigner_from_state(rho_in,
           xv, yv):
    
    #Define Wigner Function
    def w(state, xv, xy):
        xvec = xv.flatten()
        yvec = yv.flatten()
        return jnp.pi/2*dq.wigner(state, xvec = xvec, yvec = yvec, g = 2)[2]
    
    #============================================================
    #Define Preliminary Data
    dim = rho_in.dims[0]
    fac = 50
    
    #Check if dim has correct properties:
    if type(dim)!=int:
        raise('Dimension is not integer')
    
    #Compute measurement data
    w_k = w(rho_in, xv, yv)
    
    #Define Observables
    alpha = xv.flatten() + yv.flatten()*1j
    E_list = (dq.displace(dim*fac, alpha)@dq.parity(dim*fac)@dq.dag(dq.displace(dim*fac, alpha))).data[:, :dim, :dim]
    
    
    # # E_list = 1/2*(dq.eye(dim) + dq.displace(dim, alpha)@dq.parity(dim)@dq.dag(dq.displace(dim, alpha)))
    # # E_list = [1/2*(dq.eye(dim) + dq.displace(dim, alpha_k)@dq.parity(dim)@dq.dag(dq.displace(dim, alpha_k))) for alpha_k in alpha]
    #===============================================================
    #Inititalize Optimization
    
    # Define variable
    rho = cp.Variable((dim, dim), hermitian=True)
    
    # Objective
    residuals = [cp.real(cp.trace(E @ rho)) - p for E, p in zip(E_list, w_k)]
    objective = cp.Minimize(cp.sum_squares(cp.bmat(residuals)))
    
    # Constraints
    constraints = [rho >> 0, cp.trace(rho) == 1]
    
    # Solve problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose = True)
    
    # Output estimated density matrix
    rho_out = rho.value
    
    #Compute Fidelity
    fidelity = dq.fidelity(rho_in, rho_out)
    
    # res = 0
    # for E, p in zip(E_list, w_k):
    #     print((float(abs(dq.trace(E@rho_out))) - p)**2)
    #     res += (float(abs(dq.trace(E@rho_out))) - p)**2
    # print(res)
    
    return rho_out, fidelity

fock_state = dq.fock(3, 0)
# coherent_state = dq.coherent(3, 0.5)

n = 10
nx, ny = (n, n)
x = jnp.linspace(-1.5, 1.5, nx)
y = jnp.linspace(-1.5, 1.5, ny)
xv, yv = jnp.meshgrid(x, y)

rho_out, fidelity = wigner_from_state(fock_state, xv, yv)

print(fidelity)
dq.plot.wigner(fock_state)
dq.plot.wigner(rho_out)

    
        