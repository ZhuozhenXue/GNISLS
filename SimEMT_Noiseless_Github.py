# -*- coding: utf-8 -*-
"""
Created on Sun May 17 11:05:39 2026

@author: 98024
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp
import seaborn as sns
import matplotlib.pyplot as plt

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
x_true = torch.empty(0,n)

for i in range(m):
    
    # Initialize solution matrix
    solution_values = np.zeros((7, len(t_eval)))
    # Initialize derivative matrix
    derivatives_matrix = np.zeros((len(t_eval), n))  # Rows: time points, Columns: derivative of each variable
    
    # Solve ODE
    sol = solve_ivp(EMT, t_span, X0[i,:], args=(omega,b,g), t_eval=t_eval)
    
    # Extract solution values
    solution_values = sol.y
    x_true = torch.cat((x_true,torch.tensor(solution_values.T)), dim=0)


'''
####################### Scaling #######################
'''


# Compute the 99.99th percentile of each column
Q = np.percentile(x_true, 99.99, axis=0)

scales_est = Q
x_scaled = x_true/scales_est

'''
####################### Derivative estimation #######################
'''

def diff4_torch(x, h=step):
    """
    Torch-only implementation of 4th-order central difference, applied along the trajectory dimension.
    x: (T, n) single trajectory
    Returns (T, n) derivative estimates
    """
    #T = x.size(0)
    dx = torch.empty_like(x)

    # Interior points: 4th-order central difference
    dx[2:-2] = (-x[4:] + 8 * x[3:-1] - 8 * x[1:-3] + x[:-4]) / (12 * h)

    # First two points: 2nd-order forward difference
    dx[0] = (-3 * x[0] + 4 * x[1] - x[2]) / (2 * h)
    dx[1] = (-x[0] + x[2]) / (2 * h)

    # Last two points: 2nd-order backward difference
    dx[-2] = (x[-3] - x[-1]) / (2 * h)
    dx[-1] = (x[-3] - 4 * x[-2] + 3 * x[-1]) / (2 * h)

    return dx

# -------- Main workflow --------
T = len(t_eval)                    # Number of time points per trajectory
x_scaled_3d = x_scaled.reshape(m, T, n)

dx_est = torch.empty_like(x_scaled_3d)
for i in range(m):
    dx_est[i] = diff4_torch(x_scaled_3d[i])

# Flatten to match x_scaled shape
dx_est = dx_est.reshape(-1, n)

x_train = x_scaled
dx_train = dx_est



#%
'''
Start training
'''

# Loss function
# Measures the difference between two vectors
def loss_function(dx_train_i, x_train, omega_i, b_i, g_i, dim):
    # Correct usage: use brackets + 0-based indexing
    omega_i1, omega_i2, omega_i3, omega_i4, omega_i5, omega_i6, omega_i7 = omega_i
            
    x1, x2, x3 = x_true[:, 0], x_true[:, 1], x_true[:, 2]
    x4, x5, x6, x7 = x_true[:, 3], x_true[:, 4], x_true[:, 5], x_true[:, 6]
            
    W_i = b_i+omega_i1*x1+omega_i2*x2+omega_i3*x3+omega_i4*x4+omega_i5*x5+omega_i6*x6+omega_i7*x7
    
    f_pred_i = g_i*(torch.sigmoid(10 * W_i) - x_train[:, dim])
    
    return torch.mean((dx_train_i - f_pred_i) ** 2)


def gradient_descent_stls(dx_train_i, x_train, dim,
                          lr=1e-2,
                          max_outer=10,
                          max_inner=50000,
                          tolerance=1e-6,
                          threshold=0.4):
    """
    Adam optimization with STLS (Sparse Training via Library Search)
    Returns:
        omega_i, b_i, g_i   : torch.Tensor
        outer_iters         : int                   # Actual number of outer iterations
        inner_iters_record  : List[int]             # Number of inner iterations per outer round
    """

    omega_i = torch.zeros(n, requires_grad=True)
    b_i     = torch.tensor(0.0, requires_grad=True)
    g_i     = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.Adam([omega_i, b_i, g_i], lr=lr)

    mask = torch.ones_like(omega_i, dtype=torch.bool)

    inner_iters_record = []  # <-- New

    for outer in range(max_outer):
        inner_count = 0        # <-- Counter for inner iterations in this round
        for it in range(max_inner):
            inner_count += 1
            loss = loss_function(dx_train_i, x_train, omega_i, b_i, g_i, dim)

            optimizer.zero_grad()
            loss.backward()
            omega_i.grad *= mask
            optimizer.step()

            grad_norm = torch.norm(omega_i.grad)
            
            if b_i.grad is not None:
                grad_norm += torch.norm(b_i.grad)
            if g_i.grad is not None:          # New
                grad_norm += torch.norm(g_i.grad)
            if grad_norm < tolerance:
                break

        inner_iters_record.append(inner_count)  # <-- Record number of inner iterations for this round

        with torch.no_grad():
            new_mask = (torch.abs(omega_i) >= threshold)
            if torch.equal(mask, new_mask):
                print(f"STLS converged at outer={outer + 1}")
                break
            mask = new_mask
            omega_i *= mask
            optimizer = torch.optim.Adam([omega_i, b_i, g_i], lr=lr)

    outer_iters = len(inner_iters_record)  # Actual number of outer iterations
    return omega_i, b_i, g_i, outer_iters, inner_iters_record

omega_learned_stls, b_learned_stls, g_learned_stls = [], [], []
outer_counts, inner_records = [], []  # New

for i in range(n):
    w, b, g, n_outer, n_inner_list = gradient_descent_stls(
        dx_train[:, i], x_train, dim=i
    )
    omega_learned_stls.append(w)
    b_learned_stls.append(b)
    g_learned_stls.append(g)
    
    outer_counts.append(n_outer)
    inner_records.append(n_inner_list)

omega_learned_stls = torch.stack(omega_learned_stls)
b_learned_stls     = torch.tensor(b_learned_stls)
g_learned_stls     = torch.tensor(g_learned_stls)

print("omega_learned_stls:\n", omega_learned_stls)
print("b_learned_stls:\n", b_learned_stls)
print("g_learned_stls:\n", g_learned_stls)
print("outer_counts (per dim):", outer_counts)
print("inner_records (per dim):", inner_records)

