# -*- coding: utf-8 -*-
"""
Created on Sun May 17 11:15:58 2026

@author: 98024
"""

import numpy as np
from autograd import numpy as anp  
from autograd import jacobian

import torch
from scipy.integrate import solve_ivp


def EMT_Jac(X):
    x1, x2, x3, x4, x5, x6, x7 = X
    omega11, omega12, omega13, omega14, omega15, omega16, omega17 = omega[0]
    omega21, omega22, omega23, omega24, omega25, omega26, omega27 = omega[1]
    omega31, omega32, omega33, omega34, omega35, omega36, omega37 = omega[2]
    omega41, omega42, omega43, omega44, omega45, omega46, omega47 = omega[3]
    omega51, omega52, omega53, omega54, omega55, omega56, omega57 = omega[4]
    omega61, omega62, omega63, omega64, omega65, omega66, omega67 = omega[5]
    omega71, omega72, omega73, omega74, omega75, omega76, omega77 = omega[6]
    b1, b2, b3, b4, b5, b6, b7 = b[0], b[1], b[2], b[3], b[4], b[5], b[6]
    g1, g2, g3, g4, g5, g6, g7 = g[0], g[1], g[2], g[3], g[4], g[5], g[6]
    
    W1 = b1+omega11*x1+omega12*x2+omega13*x3+omega14*x4+omega15*x5+omega16*x6+omega17*x7
    W2 = b2+omega21*x1+omega22*x2+omega23*x3+omega24*x4+omega25*x5+omega26*x6+omega27*x7
    W3 = b3+omega31*x1+omega32*x2+omega33*x3+omega34*x4+omega35*x5+omega36*x6+omega37*x7
    W4 = b4+omega41*x1+omega42*x2+omega43*x3+omega44*x4+omega45*x5+omega46*x6+omega47*x7
    W5 = b5+omega51*x1+omega52*x2+omega53*x3+omega54*x4+omega55*x5+omega56*x6+omega57*x7
    W6 = b6+omega61*x1+omega62*x2+omega63*x3+omega64*x4+omega65*x5+omega66*x6+omega67*x7
    W7 = b7+omega71*x1+omega72*x2+omega73*x3+omega74*x4+omega75*x5+omega76*x6+omega77*x7
    f1 = g1*(1/(1+anp.exp(-10*W1)))
    f2 = g2*(1/(1+anp.exp(-10*W2)))
    f3 = g3*(1/(1+anp.exp(-10*W3)))
    f4 = g4*(1/(1+anp.exp(-10*W4)))
    f5 = g5*(1/(1+anp.exp(-10*W5)))
    f6 = g6*(1/(1+anp.exp(-10*W6)))
    f7 = g7*(1/(1+anp.exp(-10*W7)))
    return anp.array([f1,f2,f3,f4,f5,f6,f7])

# Set true parameter values
omega=[[0,1,0,0,0,0,0],
       [1,0,-0.6,0,0,0,0],
       [0,-0.6,0,0,0,0,0],
       [0.5,0.1,0,0,0,0,0],
       [1,1,0,0,0,0,0],
       [0,0.7,0,0,0,0,0],
       [0,-1,0,0,0,0,0]]
b=[-1,0.2,0.5,-0.5,-0.5,-0.5,0.5]
g=[1,1,1,1,1,1,1]

J_func = jacobian(EMT_Jac)

#%
def EMT(t, X, omega, b, g):
    x1, x2, x3, x4, x5, x6, x7 = X
    omega11, omega12, omega13, omega14, omega15, omega16, omega17 = omega[0]
    omega21, omega22, omega23, omega24, omega25, omega26, omega27 = omega[1]
    omega31, omega32, omega33, omega34, omega35, omega36, omega37 = omega[2]
    omega41, omega42, omega43, omega44, omega45, omega46, omega47 = omega[3]
    omega51, omega52, omega53, omega54, omega55, omega56, omega57 = omega[4]
    omega61, omega62, omega63, omega64, omega65, omega66, omega67 = omega[5]
    omega71, omega72, omega73, omega74, omega75, omega76, omega77 = omega[6]
    b1, b2, b3, b4, b5, b6, b7 = b[0], b[1], b[2], b[3], b[4], b[5], b[6]
    g1, g2, g3, g4, g5, g6, g7 = g[0], g[1], g[2], g[3], g[4], g[5], g[6]
    
    W1 = b1+omega11*x1+omega12*x2+omega13*x3+omega14*x4+omega15*x5+omega16*x6+omega17*x7
    W2 = b2+omega21*x1+omega22*x2+omega23*x3+omega24*x4+omega25*x5+omega26*x6+omega27*x7
    W3 = b3+omega31*x1+omega32*x2+omega33*x3+omega34*x4+omega35*x5+omega36*x6+omega37*x7
    W4 = b4+omega41*x1+omega42*x2+omega43*x3+omega44*x4+omega45*x5+omega46*x6+omega47*x7
    W5 = b5+omega51*x1+omega52*x2+omega53*x3+omega54*x4+omega55*x5+omega56*x6+omega57*x7
    W6 = b6+omega61*x1+omega62*x2+omega63*x3+omega64*x4+omega65*x5+omega66*x6+omega67*x7
    W7 = b7+omega71*x1+omega72*x2+omega73*x3+omega74*x4+omega75*x5+omega76*x6+omega77*x7
    f1 = g1*(1/(1+np.exp(-10*W1))-x1)
    f2 = g2*(1/(1+np.exp(-10*W2))-x2)
    f3 = g3*(1/(1+np.exp(-10*W3))-x3)
    f4 = g4*(1/(1+np.exp(-10*W4))-x4)
    f5 = g5*(1/(1+np.exp(-10*W5))-x5)
    f6 = g6*(1/(1+np.exp(-10*W6))-x6)
    f7 = g7*(1/(1+np.exp(-10*W7))-x7)
    return np.array([f1,f2,f3,f4,f5,f6,f7])

# Set true parameter values
omega=[[0,1,0,0,0,0,0],
       [1,0,-0.6,0,0,0,0],
       [0,-0.6,0,0,0,0,0],
       [0.5,0.1,0,0,0,0,0],
       [1,1,0,0,0,0,0],
       [0,0.7,0,0,0,0,0],
       [0,-1,0,0,0,0,0]]
b=[-1,0.2,0.5,-0.5,-0.5,-0.5,0.5]
g=[1,1,1,1,1,1,1]

step=0.05
end=20
t_span = (0, end)  # Time range from t=0 to t=end
# Specify time points at which to obtain the solution
t_eval = np.arange(0, end, step)

# Number of genes
n = 7
# Number of equations / trajectories
m = 1000
# Randomly generate initial conditions
X0 = np.random.rand(m,n)  # Values in [0,1]

J_list = []
for i in range(m):
    
    # Initialize solution matrix
    solution_values = np.zeros((7, len(t_eval)))
    # Initialize derivative matrix
    derivatives_matrix = np.zeros((len(t_eval), n))  # Rows: time points, Columns: derivative of each variable
    
    # Solve ODE
    sol = solve_ivp(EMT, t_span, X0[i,:], args=(omega,b,g), t_eval=t_eval)
    
    # Extract solution values
    solution_values = sol.y
    # Extract steady-state values
    x_ss = sol.y[:, -1] # Steady state of each cell
    J = J_func(x_ss)
    J_list.append(torch.from_numpy(J))

J_3D = np.stack(J_list, axis=0)  # (m, n, n)
J_m = J_3D.mean(axis=0)  # Shape (n,n)
#J_m = np.median(J_3D, axis=0)

#%
# 1. Choose a small constant c to ensure the smallest non-zero absolute value is visible in log scale
non_zero = np.abs(J_m[J_m != 0])  # 1. Remove zeros
order = np.floor(np.log10(non_zero)).min().astype(int)
c =  10.0 ** (order)          # Smaller than the smallest non-zero absolute value in the matrix

# 2. Signed logarithmic transformation
J_log = np.sign(J_m) * np.log10(1 + np.abs(J_m)/c)