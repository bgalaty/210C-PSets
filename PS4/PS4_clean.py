# Load in Packages
import numpy as np
import matplotlib.pyplot as plt
from sequence_jacobian import simple, create_model, solved

## Problem 2 - Part f
# Set Parameters
beta = 0.99
gamma = 1
varphi = 1
chi = 1
epsilon = 10
rho_a = 0.8
phi_pi = 1.5
phi_y = 0 
mu = 1/(epsilon-1)
theta_ = [0.0001, 0.25, 0.5, 0.75, 0.9999]
lambda_ = []
kappa_ = []

for i in range(len(theta_)):
    lambda_i = (1-theta_[i])*(1-beta*theta_[i])/theta_[i]
    lambda_.append(lambda_i)
    kappa_i = lambda_i*(gamma+varphi)
    kappa_.append(kappa_i)
kappa = kappa_[0]
theta = theta_[0]

@simple
def hh(c, n, varphi, gamma, beta, chi):
    wp = chi*(n**varphi)/(c**(-gamma))
    sdf = beta*(c**(-gamma))/(c(-1)**(-gamma))
    r = 1/(beta*(c**(-gamma))/(c(-1)**(-gamma)))
    return wp, sdf, r

@solved(unknowns={'f1': (1-mu)*chi/(1-theta*beta), 'f2': 1/(1-theta*beta)}, targets=['F1', 'F2'], solver="broyden_custom")
def firm(sdf, wp, a, pi, n, mu, theta, epsilon, f1, f2):    
    y = a*n
    F1 = (1 + mu) * (a * n) * (wp / a) + theta * (pi(+1) ** epsilon) * sdf(+1) * f1(+1) - f1
    F2 = (a * n) + theta * (pi(+1) ** (epsilon - 1)) * sdf(+1) * f2(+1) - f2
    return F1, F2, y

@simple
def firm2(f1, f2):
    pstar = f1 / f2
    return pstar

@simple
def central_bank(pi, v, phi_pi, beta):
    q = (1 / beta) * (pi ** phi_pi) * np.exp(v)
    return q

@simple
def mkt_clearing(y, c, pstar, q, r, pi, theta, epsilon):
    output = y - c
    inflation  = theta*(pi**(epsilon-1))+(1-theta)*(pstar**(1-epsilon)) - 1
    fisher = r(+1) - q/pi(+1)
    return fisher, output, inflation

nk = create_model([hh, firm, firm2, central_bank, mkt_clearing], name="NK")

print(nk)
print(f"Blocks: {nk.blocks}")

# steady state values
calibration = {'v': 0.0, 'a': 1.0, 'y': 1.0, 'r': 1.0/beta, 'sdf': beta, 'wp': chi, 'q': 1.0 / beta, 'gamma': gamma, 'beta': beta, 'phi_pi': phi_pi, 'kappa': kappa, 'theta': theta, 'varphi': varphi, 'chi': chi, 'mu': mu, 'epsilon': epsilon}

# solve for steady state (we know it, but running this routine helps us check for mistakes)
unknowns_ss = {'n': 1.0, 'c': 1.0, 'pi': 1.0}
targets_ss = {'fisher': 0.0, 'output': 0.0, 'inflation': 0.0}

ss = nk.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="broyden_custom")

# Print steady-state results
print("Steady-state values:")
for key, value in ss.items():
    print(f"{key}: {value}")

# Check that the residuals are close to zero
fisher_residual = ss['fisher']
output_residual = ss['output']
inflation_residual = ss['inflation']
print(f"fisher residual: {fisher_residual}")
print(f"output residual: {output_residual}")
print(f"inflation residual: {inflation_residual}")

unknowns = ['n', 'c', 'pi']
targets = ['fisher', 'output', 'inflation']
inputs = ['a', 'v']

G = nk.solve_jacobian(ss, unknowns, targets, inputs, T=300)

print(G)