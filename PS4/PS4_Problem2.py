# Load in Packages
import numpy as np
import matplotlib.pyplot as plt
from sequence_jacobian import simple, create_model

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
def hh(c, n, varphi, gamma, beta):
    wp = chi*(n**varphi)/(c**(-gamma))
    sdf = beta*(c**(-gamma))/(c(-1)**(-gamma))
    r = 1/(beta*(c**(-gamma))/(c(-1)**(-gamma)))
    return wp, sdf, r

@simple
def firm(a, pi, n, sdf, wp, theta, mu, epsilon):
    y=a*n
    f1 = (1+mu)*(a*n)*(wp/a)+theta*(pi(+1)**epsilon)*sdf(+1)*f1(+1)
    f2 = (a*n)+theta*(pi(+1)**(epsilon-1))*sdf(+1)*f2(+1)
    return y, f1, f2

@simple
def firm2(f1, f2):
    pstar = f1/f2
    return pstar

@simple
def central_bank(pi, v, phi_pi):
    q = (1/beta)*(pi**phi_pi)*np.exp(v)
    return q

@simple
def mkt_clearing(y, c, pstar, q,r,pi):
    output = y - c
    inflation  = theta*(pi**(epsilon-1))+(1-theta)*(pstar**(1-epsilon)) - 1
    fisher = r - q(-1)/(((1/theta)*(1-(1-theta)*(pstar**(1-epsilon))))**(1/(epsilon-1)))
    return fisher, output, inflation

nk = create_model([hh, firm, firm2, central_bank, mkt_clearing], name="NK")

print(nk)
print(f"Blocks: {nk.blocks}")

# steady state values
# NEED TO CALCULATE THESE
calibration = {'v': 0, 'a': 1, 'y': 0, 'r': 0, 'gamma': gamma, 'beta': beta, 'phi_pi': phi_pi, 'kappa': kappa, 'theta': theta}

# solve for steady state (we know it, but running this routine helps us check for mistakes)
unknowns_ss = {'n': 1, 'c':1, 'pi':1}
targets_ss = { "fisher": 0, "output": 0, "inflation": 0}

ss = nk.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="broyden_custom")

# checking that we are in the steady state that we expect
assert np.allclose(ss['n', 'c', 'pi'], 0)
assert np.allclose(ss['fisher'], 0)

unknowns = ['n', 'c', 'pi']
targets = ['fisher', 'output', 'inflation']
inputs = ['a', 'v']

G = nk.solve_jacobian(ss, unknowns, targets, inputs, T=300)

print(G)

calibration_copies = {}
# change calibration
for i in range(1,len(theta_)):
    calibration_i = calibration.copy()
    calibration_i['kappa'] = kappa_[i]
    calibration_i['theta'] = theta_[i]

    # calculate new steady state
    ss_i = nk.solve_steady_state(calibration_i, unknowns_ss, targets_ss, solver="broyden_custom")

    # calculate new Jacobian
    G_i = nk.solve_jacobian(ss_i, unknowns, targets, inputs, T=300)

    calibration_copies[f'G_{i}'] = G_i

T, Tplot, impact, rho, news = 300, 20, 0.01, 0.5, 10
dv = np.empty((T, 1))
dv[:, 0] = impact * rho**np.arange(T)

# plot responses for shock, nominal interest rate, real interest rate, inflation, output, and employment
plotset = ['c', 'y', 'i', 'pi', 'y', 'c']
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for i, var in enumerate(plotset):
    if var == 'v':
        irf1 = dv[:Tplot]
        irf2 = dv[:Tplot]
    else:
        irf1 = 100 * (G[var]['v'] @ dv)[:Tplot]
        irf2 = 100 * (calibration_copies['G_1'][var]['v'] @ dv)[:Tplot]
        irf3 = 100 * (calibration_copies['G_2'][var]['v'] @ dv)[:Tplot]
        irf4 = 100 * (calibration_copies['G_3'][var]['v'] @ dv)[:Tplot]
        irf5 = 100 * (calibration_copies['G_4'][var]['v'] @ dv)[:Tplot]
    axi = ax[i // 3, i % 3]
    axi.plot(irf1, label=f"theta={theta_[0]}")
    axi.plot(irf2, label=f"theta={theta_[1]}")
    axi.plot(irf3, label=f"theta={theta_[2]}")
    axi.plot(irf4, label=f"theta={theta_[3]}")
    axi.plot(irf5, label=f"theta={theta_[4]}")
    axi.set_title(f"{var}")
    axi.xlabel = "quarters"
    axi.ylabel = "% deviation"
    axi.legend()

plt.savefig(f'Documents/*School/PhD/FirstYear/210C/GitFolder/Homework/PS4/2g_IRFs.png')