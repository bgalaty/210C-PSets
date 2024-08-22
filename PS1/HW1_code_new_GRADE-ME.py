# Packages needed for this Assignment
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# Set Variables
gamma	= 1
psi 	= 1
chi		= 1
beta 	= 0.99
nu_vec = [0.25,0.5,1.00001,2,4]
count = len(nu_vec)
T = 500

# Set empty theta vector
theta_vec = []

# Calculate Corresponding Theta Values \\ Doing this arbitrarily since there's no data to calibrate with
for i in range(count):
    theta_i = 0.05
    theta_vec.append(theta_i)

fig, mx = plt.subplots(2, 2, figsize=(12, 10))

for i in range(count):
    nu = nu_vec[i]
    theta = theta_vec[i]

    ### Solve for steady state variables
    Cstar = ((1-theta)/chi * (1-theta+theta*(theta/(1-theta))**((1-nu)/nu)
                                *(1-beta)**((nu-1)/nu))**((nu-gamma)/(1-nu)))**(1/(psi+gamma))
    Mstar = ((1-beta)*((1-theta)/theta))**(-1/nu)*Cstar
    Xstar = ((1-theta)*Cstar**(1-nu)+theta*(Mstar)**(1-nu))**(1/nu)

    ## Setup Basic Model
    # Define sparse identity, above-diagonal sparse matrix, below-diagonal sparse matrix, and zero matrix
    I = sp.sparse.eye(T)
    Ip1 = sp.sparse.diags([np.ones(T-1)], [1], (T, T))
    Im1 = sp.sparse.diags([np.ones(T-1)], [-1], (T, T))
    Z = sp.sparse.csr_matrix((T, T))

    #### Market Clearing Block
    # Goods Market
    Phigmy = -I
    Phigmwp = Z
    Phigmmp = Z
    Phigmc = I
    Phigmx = Z
    Phigmq = Z

    # Money Market
    Phimmy = Z
    Phimmwp = Z
    Phimmmp = -I
    Phimmc = I
    Phimmx = Z
    Phimmq = -(1/nu)*(beta/(1-beta))*I

    dHdY = sp.sparse.bmat([[Phigmy, Phigmwp, Phigmmp, Phigmc, Phigmx, Phigmq],
                        [Phimmy, Phimmwp, Phimmmp, Phimmc, Phimmx, Phimmq]])


    assert dHdY.shape == (2*T, 6*T)

    #### Firm block
    #Production
    Phiyn = I
    Phiyp = Z
    Phiym = Z

    #Labor demand
    Phiwpn = Z
    Phiwpp = Z
    Phiwpm = Z

    #MP Identity
    Phimpn = Z
    Phimpp = -I
    Phimpm = I

    dFYdU = sp.sparse.bmat([[Phiyn, Phiyp],
                            [Phiwpn, Phiwpp],
                            [Phimpn, Phimpp]])
    dFYdZ = sp.sparse.bmat([[Phiym],
                        [Phiwpm],
                        [Phimpm]])

    assert dFYdU.shape == (3*T, 2*T)
    assert dFYdZ.shape == (3*T, 1*T)

    #### Household Block
    A = (nu - (nu - gamma)*(1-theta)*(Cstar/Xstar)**(1-nu))**(-1)

    # Consumption
    Phicn = A*Phiwpn - A*psi*I
    Phicp = A*(Phiwpp - (nu-gamma)*theta*(Mstar/(Xstar))**(1-nu)*I)
    Phicm = A*(nu-gamma)*theta*(Mstar/(Xstar))**(1-nu)*I

    dCYdU = sp.sparse.bmat([[Phicn, Phicp]])
    dCYdZ = sp.sparse.bmat([[Phicm]])

    assert dCYdU.shape == (1*T, 2*T)
    assert dCYdZ.shape == (1*T, 1*T)

    # X
    Phixn = (1-theta)*(Cstar/Xstar)**(1-nu)*Phicn 
    Phixp = (1-theta)*(Cstar/Xstar)**(1-nu)*Phicp + theta*(Mstar/(Xstar))**(1-nu)*I 
    Phixm = (1-theta)*(Cstar/Xstar)**(1-nu)*Phicm + theta*(Mstar/(Xstar))**(1-nu)*I

    dXYdU = sp.sparse.bmat([[Phixn, Phixp]])
    dXYdZ = sp.sparse.bmat([[Phixm]])

    assert dXYdU.shape == (1*T, 2*T)
    assert dXYdZ.shape == (1*T, 1*T)

    ### Bonds Block
    Phiqn = -nu*(Phicn*I - Phicn*Ip1) + (nu-gamma)*(Phixn*I - Phixn*Ip1)
    Phiqp = -nu*Phicp*(I - Ip1) - (I - Ip1) + (nu - gamma)*(Phixp*I - Phixp*Ip1)
    Phiqm = -nu*(Phicm*I - Phicm*Ip1) + (nu-gamma)*(Phixm*I - Phixm*Ip1)

    dBYdU = sp.sparse.bmat([[Phiqn, Phiqp]])
    dBYdZ = sp.sparse.bmat([[Phiqm]])
    assert dBYdU.shape == (1*T, 2*T)
    assert dBYdZ.shape == (1*T, 1*T)

    # stack to get dYdU
    dYdU = sp.sparse.bmat([[dFYdU],
                        [dCYdU],
                        [dXYdU],
                        [dBYdU]])

    # stack to get dYdZ
    dYdZ = sp.sparse.bmat([[dFYdZ],
                        [dCYdZ],
                        [dXYdZ],
                        [dBYdZ]])

    assert dYdU.shape == (6*T, 2*T)
    assert dYdZ.shape == (6*T, 1*T)

    # compute dHdU using the chain rule dHdU = dHdY @ dYdU (@ is the python matrix multiplication operator)
    dHdU = dHdY @ dYdU

    # compute dHdZ using the chain rule dHdZ = dHdY @ dYdZ (@ is the python matrix multiplication operator)
    dHdZ = dHdY @ dYdZ

    assert sp.sparse.issparse(dHdZ) == True
    assert sp.sparse.issparse(dHdU) == True

    assert dHdU.shape == (2*T, 2*T)
    assert dHdZ.shape == (2*T, 1*T)

    # compute the Jacobian of the model
    dUdZ = - sp.sparse.linalg.spsolve(dHdU, dHdZ)
    dYdZ = dYdU @ dUdZ + dYdZ

    dXdZ = sp.sparse.bmat([[dUdZ],
                           [dYdZ]])
        
    assert dUdZ.shape == (2*T, T)
    assert dYdZ.shape == (6*T, T)
    assert dXdZ.shape == (8*T, T)

    ## Plotting IRFs
    # plot IRFs to Money Supply shock with persistence rho
    rho_m   = 0.99
    m = np.zeros((T, 1))
    m[0] = 1
    for t in range(1, T):
        m[t] = rho_m * m[t-1]

    # compute impulse response functions
    X = dXdZ @ m

    # unpack X into its components y, wp, c, q
    n = X[0:T]
    p = X[T:2*T]
    y = X[2*T:3*T]
    wp = X[3*T:4*T]
    mp = X[4*T:5*T]
    c = X[5*T:6*T]
    x = X[6*T:7*T]
    q = X[7*T:8*T]

    # plot impulse response functions
    # fig, mx = plt.subplots(2, 2, figsize=(12, 10))

    mx[0, 0].plot(m, label='m')
    mx[0, 0].set_title('Money Supply')
    mx[0, 1].plot(c, label='c')
    mx[0, 1].set_title('Consumption')
    mx[1, 0].plot(p, label='p')
    mx[1, 0].set_title('Prices')
    mx[1, 1].plot(q, label='q')
    mx[1, 1].set_title('Nominal Interest Rate')
    # plt.figtext(0.5, 0.95, f"Nu is {nu}", wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(f'Documents/*School/PhD/FirstYear/210C/GitFolder/Homework/PS1/IRFs_{i}.png')

plt.savefig(f'Documents/*School/PhD/FirstYear/210C/GitFolder/Homework/PS1/IRFs.png')