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
a = [a0]
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
one_b[2, 1].set_title('Interest Rate')
one_b[3, 0].plot(r, label='r')
one_b[3, 0].set_title('Real Returns')
one_b[3, 1].plot(n, label='n')
one_b[3, 1].set_title('Employment')


plt.savefig(f'Documents/*School/PhD/FirstYear/210C/GitFolder/Homework/PS4/1b_IRFs.png')

## Problem 1 - Part d

from sequence_jacobian import simple, create_model, solved

@simple
def nkpc(pi, beta, kappa, phi_pi, gamma, a):
    yflex = ((1 + phi_pi) * a) / (gamma + phi_pi)
    y = yflex + 1 / kappa * (pi - beta * pi(+1))
    yyflexdiff = y - yflex
    n = y - a
    return y, yflex, yyflexdiff, n

@simple
def central_bank(pi, phi_pi):
    i = phi_pi * pi
    return i

@simple
def mkt_clearing(y, i, pi, gamma):
    euler = gamma * y + i - pi(+1) - gamma * y(+1)
    r = i - pi(+1)
    c = y
    return euler, r, c

print(nkpc)
print(f"Inputs: {nkpc.inputs}")
print(f"Outputs: {nkpc.outputs}")

nk = create_model([nkpc, central_bank, mkt_clearing], name="NK")

print(nk)
print(f"Blocks: {nk.blocks}")

unknowns = ['pi']
targets = ['euler']
inputs = ['a']

# steady state values
calibration = {'yflex': 0, 'a': 0, 'y': 0, 'c': 0, 'r': 0, 'i': 0, 'gamma': gamma, 'beta': beta, 'phi_pi': phi_pi, 'kappa': kappa}

# solve for steady state (we know it, but running this routine helps us check for mistakes)
unknowns_ss = {'pi': 0}
targets_ss = { "euler": 0}

ss = nk.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="broyden_custom")

# checking that we are in the steady state that we expect
assert np.allclose(ss['pi'], 0)
assert np.allclose(ss['euler'], 0)


G = nk.solve_jacobian(ss, unknowns, targets, inputs, T=300)

print(G)

T, Tplot, impact, rho, news = 300, 20, 0.01, 0.8, 10
da = np.empty((T, 1))
da[:, 0] = impact * rho**np.arange(T)


# plot responses=
plotset = ['a', 'y', 'pi', 'yflex', 'yyflexdiff', 'i', 'r', 'n']
fig, ax = plt.subplots(4, 2, figsize=(12, 20))
for i, var in enumerate(plotset):
    if var == 'a':
        irf1 = da[:Tplot]
    else:
        irf1 = 100 * (G[var]['a'] @ da)[:Tplot]
    axi = ax[i // 2, i % 2]
    ax[0, 0].set_title('Productivity')
    ax[0, 1].set_title('Output')
    ax[1, 0].set_title('Inflation')
    ax[1, 1].set_title('y_flex')
    ax[2, 0].set_title('Output Gap')
    ax[2, 1].set_title('Interest Rate')
    ax[3, 0].set_title('Real Returns')
    ax[3, 1].set_title('Employment')
    axi.plot(irf1, label="kappa=0.1")

plt.savefig(f'Documents/*School/PhD/FirstYear/210C/GitFolder/Homework/PS4/1d_IRFs.png')