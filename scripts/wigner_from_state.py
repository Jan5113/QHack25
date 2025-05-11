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
           x_vec, y_vec):
    
    xv, yv = jnp.meshgrid(x_vec, y_vec)
    #Define Wigner Function
    def w(state, xv, xy):
        return jnp.pi/2*dq.wigner(state, xmax = xv.max(), ymax = yv.max(), npixels = len(xv))[2]
    
    #============================================================
    #Define Preliminary Data
    dim = rho_in.dims[0]
    fac = 10
    
    #Check if dim has correct properties:
    if type(dim)!=int:
        raise('Dimension is not integer')
    
    #Compute measurement data
    w_k = w(rho_in, xv, yv).flatten()
    
    #Define Observables
    alpha = (xv.T + yv.T*1j).flatten()
    E_list = (dq.displace(dim*fac, alpha)@dq.parity(dim*fac)@dq.dag(dq.displace(dim*fac, alpha))).data[:, :dim, :dim]
    
    #===============================================================
    #Inititalize Optimization
    
    # Define variable
    rho = cp.Variable((dim, dim), hermitian=True)
    
    # Objective
    exprs = cp.hstack([cp.real(cp.trace(E@rho)) - k for E, k in zip(E_list, w_k)])
    objective = cp.Minimize(cp.sum_squares(exprs))
    # print(objective)
    
    # Constraints
    constraints = [rho >> 0, cp.trace(rho) == 1]
    
    # Solve problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose = True)
    
    # Output estimated density matrix
    rho_out = rho.value
    #Compute Fidelity
    fidelity = dq.fidelity(rho_in, rho_out)
    
    return rho_out, fidelity
    
        