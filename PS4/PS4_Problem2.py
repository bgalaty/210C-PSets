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

# @simple
# def firm(a, pi, n, sdf, wp, theta, mu, epsilon):
#     y=a*n
#     f1 = (1+mu)*(a*n)*(wp/a)+theta*(pi(+1)**epsilon)*sdf(+1)*f1(+1)
#     f2 = (a*n)+theta*(pi(+1)**(epsilon-1))*sdf(+1)*f2(+1)
#     # f1 = (1+mu)*(a*n)*(wp/a)+theta*(pi(+1)**epsilon)*sdf(+1)*(1+mu)*(a*n)*(wp/a)
#     # f2 = (a*n)+theta*(pi(+1)**(epsilon-1))*sdf(+1)*(a*n)
#     return y, f1, f2

# @solved(unknowns={'f1': float(1),'f2':float(1)}, targets=['F1','F2'], solver="broyden_custom")
# def firm(sdf, wp, a, pi, n, mu, theta, epsilon, f1, f2):    
 
#     F1 = (1+mu)*(a*n)*(wp/a)+theta*(pi(+1)**epsilon)*sdf(+1)*f1(+1)-f1
#     F2 = (a*n)+theta*(pi(+1)**(epsilon-1))*sdf(+1)*f2(+1)-f2
#     return f1,f2

# @simple
# def firm2(f1, f2):
#     pstar = f1/f2
#     return pstar

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

# @simple
# def central_bank(pi, v, phi_pi, beta):
#     q = (1/beta)*(pi**phi_pi)*np.exp(v)
#     return q

@simple
def central_bank(pi, v, phi_pi, beta):
    q = (1 / beta) * (pi ** phi_pi) * np.exp(v)
    return q

@simple
def mkt_clearing(y, c, pstar, q, r, pi, theta, epsilon):
    output = y - c
    inflation  = theta*(pi**(epsilon-1))+(1-theta)*(pstar**(1-epsilon)) - 1
    fisher = r(+1) - q/pi(+1)
    # fisher = r - q(-1)/(((1/theta)*(1-(1-theta)*(abs(pstar)**(1-epsilon))))**(1/(epsilon-1)))
    return fisher, output, inflation

nk = create_model([hh, firm, firm2, central_bank, mkt_clearing], name="NK")

print(nk)
print(f"Blocks: {nk.blocks}")

# steady state values
# NEED TO CALCULATE THESE
# calibration = {'v': 0, 'a': 1, 'y': 1, 'r': 1, 'sdf': 1, 'wp': chi,'q':1/beta ,'gamma': gamma, 'beta': beta, 'phi_pi': phi_pi, 'kappa': kappa, 'theta': theta, 'varphi': varphi, 'chi': chi, 'mu':mu, 'epsilon':epsilon}

# # solve for steady state
# unknowns_ss = {'n': 1, 'c':1, 'pi':1}
# targets_ss = { "fisher": 0, "output": 0, "inflation": 0}

# steady state values
calibration = {'v': 0.0, 'a': 1.0, 'y': 1.0, 'r': 1.0/beta, 'sdf': beta, 'wp': chi, 'q': 1.0 / beta, 'gamma': gamma, 'beta': beta, 'phi_pi': phi_pi, 'kappa': kappa, 'theta': theta, 'varphi': varphi, 'chi': chi, 'mu': mu, 'epsilon': epsilon}

# solve for steady state (we know it, but running this routine helps us check for mistakes)
unknowns_ss = {'n': 0.9, 'c': 1.0, 'pi': 1.0}
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

# checking that we are in the steady state that we expect
# Check that the unknowns are close to their target values
assert np.allclose(ss['n'], 0.9), f"n is not close to 1.0, but {ss['n']}"
assert np.allclose(ss['c'], 1.0), f"c is not close to 1.0, but {ss['c']}"
assert np.allclose(ss['pi'], 1.0), f"pi is not close to 1.0, but {ss['pi']}"
assert np.allclose(fisher_residual, 0.0), f"fisher residual is not close to 0.0, but {fisher_residual}"
assert np.allclose(output_residual, 0.0), f"output residual is not close to 0.0, but {output_residual}"
assert np.allclose(inflation_residual, 0.0), f"inflation residual is not close to 0.0, but {inflation_residual}"

unknowns = ['n', 'c', 'pi']
targets = ['fisher', 'output', 'inflation']
inputs = ['a', 'v']

G = nk.solve_jacobian(ss, unknowns, targets, inputs, T=300)

print(G)

# calibration_copies = {}
# # change calibration
# for i in range(1,len(theta_)):
#     calibration_i = calibration.copy()
#     calibration_i['kappa'] = kappa_[i]
#     calibration_i['theta'] = theta_[i]

#     # calculate new steady state
#     ss_i = nk.solve_steady_state(calibration_i, unknowns_ss, targets_ss, solver="broyden_custom")

#     # calculate new Jacobian
#     G_i = nk.solve_jacobian(ss_i, unknowns, targets, inputs, T=300)

#     calibration_copies[f'G_{i}'] = G_i

# T, Tplot, impact, rho, news = 300, 20, 0.01, 0.5, 10
# dv = np.empty((T, 1))
# dv[:, 0] = impact * rho**np.arange(T)

# # plot responses for shock, nominal interest rate, real interest rate, inflation, output, and employment
# plotset = ['c', 'y', 'i', 'pi', 'y', 'c']
# fig, ax = plt.subplots(2, 3, figsize=(15, 10))
# for i, var in enumerate(plotset):
#     if var == 'v':
#         irf1 = dv[:Tplot]
#         irf2 = dv[:Tplot]
#     else:
#         irf1 = 100 * (G[var]['v'] @ dv)[:Tplot]
#         irf2 = 100 * (calibration_copies['G_1'][var]['v'] @ dv)[:Tplot]
#         irf3 = 100 * (calibration_copies['G_2'][var]['v'] @ dv)[:Tplot]
#         irf4 = 100 * (calibration_copies['G_3'][var]['v'] @ dv)[:Tplot]
#         irf5 = 100 * (calibration_copies['G_4'][var]['v'] @ dv)[:Tplot]
#     axi = ax[i // 3, i % 3]
#     axi.plot(irf1, label=f"theta={theta_[0]}")
#     axi.plot(irf2, label=f"theta={theta_[1]}")
#     axi.plot(irf3, label=f"theta={theta_[2]}")
#     axi.plot(irf4, label=f"theta={theta_[3]}")
#     axi.plot(irf5, label=f"theta={theta_[4]}")
#     axi.set_title(f"{var}")
#     axi.xlabel = "quarters"
#     axi.ylabel = "% deviation"
#     axi.legend()

# plt.savefig(f'Documents/*School/PhD/FirstYear/210C/GitFolder/Homework/PS4/2g_IRFs.png')