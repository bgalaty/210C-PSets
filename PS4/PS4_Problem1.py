# Load in Packages
import numpy as np
import matplotlib.pyplot as plt

## Problem 1 - Part b
# Set Parameters
beta = 0.99
sigma = 1
kappa = 0.1
rho = 0.8
phi_pi = 1.5
gamma = 1
a0 = 1

# Set-up
eta_y = sigma*kappa*(phi_pi-rho)/((1-beta*rho)*(1-rho)+sigma*kappa*(phi_pi-rho))
eta_pi = kappa*(eta_y - 1)/(1-beta*rho)
eta_i = phi_pi*eta_pi

# Solve for IRF
a_t1 = a0
i_t1 = 0
a = []
y = []
pi = []
yflex = []
y_yflex = []
i = []
r = []
n = []

for j in range(20):
    # epsilon = np.random.normal(loc=0.0, scale=1.0, size=None)
    epsilon = 0
    a_t = rho*a_t1 + epsilon
    a.append(a_t)
    y_t = eta_y*a_t
    y.append(y_t)
    pi_t = eta_pi*a_t
    pi.append(pi_t)
    yflex_t = a_t
    yflex.append(yflex_t)
    y_yflex_t = y_t-yflex_t
    y_yflex.append(y_yflex_t)
    i_t = eta_i*a_t
    i.append(i_t)
    r_t1 = i_t1 - pi_t
    r.append(r_t1)
    n_t = y_t - a_t
    n.append(n_t)
    a_t1 = a_t
    i_t1 = i_t

r.pop(0)

fig, one_b = plt.subplots(4, 2, figsize=(12, 20))
one_b[0, 0].plot(a, label='a')
one_b[0, 0].set_title('Productivity')
one_b[0, 1].plot(y, label='y')
one_b[0, 1].set_title('Output')
one_b[1, 0].plot(pi, label='pi')
one_b[1, 0].set_title('Inflation')
one_b[1, 1].plot(yflex, label='yflex')
one_b[1, 1].set_title('y_flex')
one_b[2, 0].plot(y_yflex, label='y_yflex')
one_b[2, 0].set_title('Output Gap')
one_b[2, 1].plot(i, label='i')
one_b[2, 1].set_title('Investment')
one_b[3, 0].plot(r, label='r')
one_b[3, 0].set_title('Real Returns')
one_b[3, 1].plot(n, label='n')
one_b[3, 1].set_title('Employment')


plt.savefig(f'Documents/*School/PhD/FirstYear/210C/GitFolder/Homework/PS4/1b_IRFs.png')