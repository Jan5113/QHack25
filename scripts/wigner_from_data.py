# -*- coding: utf-8 -*-
"""
Created on Sun May 11 01:46:39 2025

@author: Somesh
"""

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


def wigner_from_data(wigner_measurements,
                     xvec, yvec,
                     dim):
    
    #============================================================
    #Define Preliminary Data
    fac = 5
    xv, yv = jnp.meshgrid(xvec, yvec)
    
    #Compute measurement data
    w_k = jnp.pi/2*wigner_measurements.flatten()
    
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
    
    return rho_out

state = dq.fock(5, 1)
data = dq.wigner(state, xmax = 2, ymax = 2, npixels = 10)
rho_out = wigner_from_data(data[2], data[0], data[1], 5)
fidelity = dq.fidelity(state, rho_out)


    
        