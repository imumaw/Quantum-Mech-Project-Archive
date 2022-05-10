#Tianyi Wang and Isaiah Mumaw

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

##############################################################################
#metropolis-hastings integration (redesigned specifically for this project)
##############################################################################

def metropolis_int(hamiltonian, wave_func, dim, size=100000, scale=1, acceptance=True, **kwargs):
    """Generates points based on Metropolis-Hastings algorithm
    
    Parameters:
        hamiltonian: hamiltonian function (being integrated), with proper modifications as shown in class notes
        wave_func (func): wave function. Square of this gives distribution function
        dim (int): number of dimensions over which to integrate. 
            For atomic simulations, dim = 3N, where N is the number of electrons
        size (int, optional): number of points to generate (defaults 100000)
        acceptance (bool, optional): if True, prints the acceptance rate at the end of the calculation
        **kwargs (optional): additional arguments used in the Hamiltonian
        
    Returns:
        integral (number): energy of system
        accept_rate (number): acceptance rate of algorithm
    """
    
    #set up variables
    rand_vals = np.random.rand(size)   #between 0 and 1, compare to p_move
    rand_chain = np.random.normal(loc=0,scale=scale,size=(dim,size)) #generate steps
    mean = 0
    naccept = 0
    
    for j in range(1,size):

        #get current and proposed coordinates
        r1 = rand_chain[:,j-1]
        r2 = r1 + rand_chain[:,j]
        
        #get probabilities at each coordinate
        p1 = wave_func(r1,**kwargs)**2
        p2 = wave_func(r2,**kwargs)**2
        
        #probability of moving
        p_move = p2 / p1
        
        #determine if move happens
        if rand_vals[j] < p_move:
            rand_chain[:,j]=r2
            naccept+=1
        else:
            rand_chain[:,j]=r1
        
        mean += hamiltonian(rand_chain[:,j],**kwargs)
    
    #return results
    if acceptance:
        accept_rate = naccept/size
        return mean/size, accept_rate
    else:
        return mean/size

##############################################################################
#trial wavefunctions for various setups
##############################################################################

def hydrogen(r_e,a,z=0):
    r = np.linalg.norm(r_e)
    return np.exp(-r/a)

def helium(grid,a,z=0):
    r_e1 = grid[0:3]
    r_e2 = grid[3:6]
    return hydrogen(r_e1,a) * hydrogen(r_e2,a)

def dihydrogen(grid,a,R,z=0):
    r_e1 = grid[0:3]
    r_e2 = grid[3:6]
    R_vect = np.array([R,0,0])    
    r_1 = np.linalg.norm(r_e1)
    return hydrogen(r_1,a) * hydrogen(np.linalg.norm(r_e2-R_vect),a)

##############################################################################
#Laplacian coefficient (true for all trial wavefunctions with form similar to hydrogen)
##############################################################################

def laplace_coeff(r,a):
    return (r-2*a)/(r*a**2)

##############################################################################
#Hamiltonians
##############################################################################

def H_mono(grid,z=1,a=1,e_charge=1,hbar=1,eps0=(1/(4*np.pi)),m=1):
    """Hamiltonian of a monoatomic setup, divided by the trial wave function.
    Number of electrons is inferred based on grid size.
    
    Parameters:
        grid (3N-dimensional array): positions of N electrons. Formatted as [x1,y1,z1,x2,y2,z2,...]
        a (number, optional): minimizing constant
        z (positive int, optional): number of protons
        e_charge, hbar, eps0, m (numbers, optional): constants involved in calculation
    """
    
    #set up variables
    num_electrons = int(len(grid)/3)
    r_e_array = np.zeros((num_electrons,3))
    v = 0
    k = 0
    
    #get energy values for each isolated electron
    for i in range(num_electrons):
        r_e = grid[3*i:3*(i+1)]
        r_e_array[i] = r_e
        r = np.linalg.norm(r_e)
        
        #get kinetic and potential energy of each electron
        v += -(z*e_charge**2)/(4*np.pi*eps0*r)
        k += -hbar**2/(2*m) * laplace_coeff(r,a)
    
    #get interactions between electrons
    for i in range(num_electrons):
        j = i+1
        
        while j <= (num_electrons-1):
            v += (e_charge**2)/(4*np.pi*eps0*np.linalg.norm(r_e_array[i]-r_e_array[j]))
            j+=1
    
    return k+v

##############################################################################

def H_dihydrogen(grid,z=1,e_charge=1,hbar=1,eps0=(1./(4*np.pi)),m=1,a=1,R=1):
    """Finds result of Hpsi/psi on trial wave function for two 
    electrons at a specific point.
    
    Parameters:
        grid (6D array): positions of both electrons. Formatted as [x1,y1,z1,x2,y2,z2,Rx]
        R (number): distance between nuclei
        a (number): minimizing constant
        z (positive int, optional): number of protons
        e_charge, hbar, eps0, m (numbers, optional): constants involved in calculation
    """
    
    re1 = grid[0:3]
    re2 = grid[3:6]
    Rvect = np.array([R,0,0])
    
    r1 = np.linalg.norm(re1)
    r2 = np.linalg.norm(re2)
    
    #get potential energies
    vr1 = -(z*e_charge**2)/(4*np.pi*eps0*r1)
    vr2 = -(z*e_charge**2)/(4*np.pi*eps0*r2)
    vr1R = -(z*e_charge**2)/(4*np.pi*eps0*np.linalg.norm(re1-Rvect))
    vr2R = -(z*e_charge**2)/(4*np.pi*eps0*np.linalg.norm(re2-Rvect))
    vR = (z*e_charge**2)/(4*np.pi*eps0*R)
    vr1r2 = (e_charge**2)/(4*np.pi*eps0*np.linalg.norm(re1-re2))
    
    v = vr1 + vr2 + vr1R + vr2R + vR + vr1r2
    
    #alternative grid for symmetry term with r1 and r2 flipped
    grid_alt = np.append(re2,re1,axis=0)
    
    #get kinetic energies
    k11 = -hbar**2/(2*m) * laplace_coeff(r1,a)*dihydrogen(grid,a,R)
    k12 = -hbar**2/(2*m) * laplace_coeff(np.linalg.norm(re2-Rvect),a)*dihydrogen(grid,a,R)
    k21 = -hbar**2/(2*m) * laplace_coeff(np.linalg.norm(re1-Rvect),a)*dihydrogen(grid_alt,a,R)
    k22 = -hbar**2/(2*m) * laplace_coeff(r2,a)*dihydrogen(grid_alt,a,R)
    
    k = (k11+k12+k21+k22)/(dihydrogen(grid,a,R) + dihydrogen(grid_alt,a,R))
    
    return k+v

##############################################################################
#rotational and vibrational energies
##############################################################################

def get_w_vib(R,E,m=1836):
    """Finds the value of w for vibrational case, 0.5*m*w^2*(R-R_ground)^2 - E_ground, which best fits calculated energy for diatomic molecule
    
    Parameters:
        R (1D array): nuclear distances
        E (1D array): array of energies based on nuclear distance, already minimized for "a"
        m (number, optional): nuclear mass
    """
    
    ground_index = np.argmin(E)
    E_ground = E[ground_index]
    R_ground = R[ground_index]
    
    #reset plot such that ground energy is centered at (0,0)
    E_norm = E-E_ground
    R_norm = R-R_ground
    
    polynomial = np.polynomial.polynomial.polyfit(R_norm,E_norm,[2])
    
    m_red = m/2
    w = np.sqrt(2*polynomial[2]/m_red)
    
    return w


def T_vibrational(R,E,hbar=1,kb=1,m=1836):
    """Finds the temperature at which vibrational energy contributes to overall energy in diatomic molecule
    
    Parameters:
        R (1D array): nuclear distances
        E (1D array): array of energies based on nuclear distance, already minimized for "a"
        hbar, kb (numbers, optional): constants used in calculation. Defaults to natural units
        m (number, optional): nuclear mass
        
    Returns:
        T, w, E (numbers): Temperature, angular frequency, and harmonic oscillator energy, respectively
    """
    
    #harmonic oscillator
    w = get_w_vib(R,E,m=m)
    E_harmonic = 0.5*hbar*w
    
    #energy is given by E=kb*T, where kinetic and potential energies each contribute 1/2
    #E_harmonic is the total kinetic and potential energy
    T = E_harmonic/kb
    
    return T, w, E_harmonic

##############################################################################

def T_rotational(R,E,hbar=1,kb=1,m=1836):
    """Finds the energy and temperature at which rotational energies become active in diatomic molecule
    
    Parameters:
        R (1D array): nuclear distances
        E (1D array): array of energies based on nuclear distance, already minimized for "a"
        hbar, kb (numbers, optional): constants used in calculation. Defaults to natural units
        m (number, optional): nuclear mass (defaults to 1836, proton mass when m_electron=1)
        
    Returns:
        T, E (numbers): Temperature and rotational energy, respectively
    """
    
    ground_index = np.argmin(E)
    R_ground = R[ground_index]
    
    #get energies
    m_red = m/2
    E_rotational = 2*hbar**2/(2*m_red*R_ground**2)
    
    #2 degrees of freedom
    T = E_rotational/kb
    
    return T, E_rotational


##############################################################################
#visualization
##############################################################################

def energy_curve(a,E,title="Energy vs. minimization constant",xlabel="Minimization constant",ylabel="Energy (atomic units)"):
    """Plots energy with respect to minimization constant. Can also just do general 2D plots.
    
    Parameters:
        a (array): minimization constants
        E (array): energies at given minization constants
        title, xlabel, ylabel (strings, optional): plotting data
    """
    plt.plot(a,E)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return

##############################################################################

def energy_heatmap(a,R,E,power=False,title="Energy vs. minimization and nuclear distance",xlabel="Minimization constant",ylabel="Nuclear distance",cbarlabel="Energy",cmap=plt.cm.coolwarm,shading="gouraud"):
    """Plots energy with respect to minimization constant (though can also plot with respect to R if properly done)
    
    Parameters:
        a (1D array): minimization constants
        R (1D array): distances between nuclei
        E (2D array): energies at given minization constants.
        power (bool, optional): gives the option of making color bar based on a power law
        title, xlabel, ylabel, cbarlabel, cmap, shading (strings, optional): plotting data
    """
    
    if power:
        plot = plt.pcolormesh(a,R,E,norm=colors.PowerNorm(gamma=0.3),cmap=cmap,shading=shading)
    else:
        plot = plt.pcolormesh(a,R,E,cmap=cmap,shading=shading)

    cbar = plt.colorbar(plot,shrink=.9,pad=0.1)
    cbar.set_label("Energy",rotation=270)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return
    
    
