"""Isaiah Mumaw and Dan Palmer
University of Notre Dame


HOW TO USE:
    
Get stationary states:
1) Generate Hamiltonian for given basis states and potential energy using hmatrix()
    -includes progress bar, however this is pretty glitchy when running inside Spyder. Use terminal for best results
2) Diagonalize using diagonalize(). This returns the new matrix, plus all eigenvalues/vectors
3) Get the stationary state at energy level n using s_state()
    -includes plotting option
    
Test for convergence:
Can either use convergence_test() or plot_convergence_test(). Both will return the energies at each 
step, the dimension of the Hamiltonian at each step, the analyzed energy levels, and where the
highest energy level converged (if it converged at all). The function plot_convergence_test() will
also plot the energies vs the dimension of the array.
"""

import scipy.special as spspec
import scipy.integrate as spint
import math
import numpy as np
import matplotlib.pyplot as plt
import time

###################################################################################################

def ho(x,n,constants):
    """Calculates wave function of a harmonic oscillator. Returns a 1D array.

    Parameters:
        x (number/array): points or points to evaluate at
        n (int): energy level
        constants (dictionary): any additional constants needed (only HBAR, M, W will be used)
    """
    
    #calculate leading constants
    const = ((constants["M"]*constants["W"])/(math.pi*constants["HBAR"]))**.25
    nconst = 1/(np.sqrt((2.0**n)*math.factorial(n)))

    #convert x to xi and calculate the other components
    xi = x*np.sqrt(constants["M"]*constants["W"]/constants["HBAR"])
    nherm = spspec.eval_hermite(n,xi)
    expo = np.exp((-xi**2.0)/2.0)

    #multiply all components
    return const*nconst*nherm*expo

###################################################################################################

def isorthogonal(vec1, vec2, tolerance=1e-12):
    """"Checks for orthogonality between two vectors of numpy array type. Must be of equal dimension. Primarily used for testing purposes.

    Tolerance defaults to 1e-12. May need to be increased for low resolutions.
    """

    dotprod = abs(np.dot(vec1, vec2))

    if dotprod <= tolerance:
        return True
    else:
        return False

###################################################################################################

def gen_hexpect(x,vfunc,wave_func,n1,n2,constants):
    """Calculates expectation value of the Hamiltonian for the harmonic oscillator

    Parameters:
        x (array): set of points over which wave function is calculated
        vfunc (function): potential energy function. Automatically passes w, m, hbar to this function.
        wave_func (function): stationary state wave function
        n1, n2 (int): energy levels of the wave functions. n1 is the bra (and row), n2 is the ket (and column)
        constants (dictionary): any additional constants needed
    """
    #get wave functions
    psi_n1 = wave_func(x,n1,constants)
    psi_n1 = np.transpose(psi_n1)
    psi_n2 = wave_func(x,n2,constants)

    #potential energy integral
    p_integrand = psi_n1 * vfunc(x,constants) * psi_n2
    p_int = spint.simps(p_integrand,x=x)

    #kinetic energy integral
    d1 = np.gradient(psi_n2,x)
    d2 = np.gradient(d1,x)
    t_integrand = psi_n1 * -0.5*constants["HBAR"]**2/constants["M"] * d2
    t_int = spint.simps(t_integrand,x=x)

    return p_int + t_int

###################################################################################################

def hmatrix(x,vfunc,wave_func,constants,nmax=50,progress=False):
    """Creates full matrix for the Hamiltonian of a harmonic oscillator

    Parameters:
        x (array): set of points over which wave function is calculated
        vfunc (function): potential energy function
        wave_func (function): stationary state wave function
        constants (dictionary): any additional constants needed
        nmax (int): maximum dimension of the square matrix. Default 50.
        progress (bool): when true, keeps a running progress meter. Can be helpful for larger matrices. Defaults True
    """
    
    #create empty matrix
    hamiltonian = np.empty((nmax,nmax))
    i,j = 0,0
    
    #creating progress meter
    if progress:
        start = time.perf_counter()
        print("Writing Hamiltonian:")
        current_cells = 0
        total_cells = nmax**2

    #filling hamiltonian
    for i in range(len(hamiltonian)):
        for j in range(len(hamiltonian[0])):
            
            #moving progress meter
            if progress:
                bar = math.floor(40*(current_cells/total_cells))
                print(bar*"█"+(40-bar)*"-"+"  "+str(round(100*current_cells/total_cells,1))+"%", end="\r")
                current_cells += 1
            
            #getting value at current position of matrix
            hamiltonian[i,j] = gen_hexpect(x,vfunc,wave_func,i,j,constants)
       
    #finishing progress meter
    if progress:
        print(40*"█"+" 100.0%")
        print("Runtime: " +str(round(time.perf_counter()-start,2))+" s")

    return hamiltonian

###################################################################################################

def diagonalize(matrix):
    """Diagonalizes a given matrix, in the format of a numpy array
    
    Returns the diagonalized matrix as well as the associated eigenvalues/eigenvectors"""

    #getting eigenvals/vecs
    eigenvals, eigenvecs = np.linalg.eigh(matrix)
    
    #creating empty Hamiltonian
    shape = np.shape(matrix)
    new_mat = np.zeros(shape)
    
    #creating diagonal matrix
    for i in range(shape[0]):
        new_mat[i][i] = eigenvals[i]
        
    return new_mat,eigenvals,eigenvecs

###################################################################################################

def s_state(x,eigenvecs,wave_func,n,constants,plot=False):
    """Creates linear combination of stationary states. Returns resulting function
    
    Parameters:
        x (array): set of points over which wave function is calculated
        eigenvecs (array): array of eigenvectors as given from diagonalize()
        wave_func (function): stationary state wave function
        n (int): specific eigenvectors to access
        plot (bool, optional): plots the state
    """

    #set up variables
    max_n=len(eigenvecs[0])
    stationary_state=np.zeros(len(x))
    
    #setting up linear combination
    for i in range(max_n):
        stationary_state+=eigenvecs[i,n]*wave_func(x,i,constants)
        
    #plotting
    if plot:
        plt.plot(x,stationary_state)
        
    return stationary_state

###################################################################################################

def convergence_test(x,v_func,wave_func,nmin,nmax,constants,tolerance=1e-3,test_range=5):
    """Tests for energy level convergence. 
    
    Returns requested energies calculated at each Hamiltonian size, 
    as well as the levels they were found at and the total dimensions calculated for each test
    
    Parameters:
        x (array): set of points over which wave function is calculated
        v_func (function): potential energy function
        wave_func (function): stationary state wave function
        nmin, nmax (int): minimum and maximum energy levels to analyze. nmax must be greater than or equal to nmin
        tolerance (float): maximum difference between calculations for convergence to be true
        test_range (int): minimum number of tests to perform. Defaults to 5
    
    Returns:
        energies (2D array): energy of each level at each step
        dimensions (1D array): rows (or columns) of Hamiltonian array at each step
        levels (1D array): energy levels being analyzed
        convergence (int): dimension at which highest analyzed energy converged. If 0, did not converge
    """

    #set up arrays
    energies = np.zeros((nmax-nmin+1,test_range))
    dimensions = np.arange(nmax+1,nmax+test_range+1)    #size of Hamiltonian for each test
    levels = np.arange(nmin,nmax+1)                   #specific energy levels to test
    
    for j,n in enumerate(dimensions):
        
        #create diagonalized hamiltonian 
        hamiltonian = hmatrix(x,v_func,wave_func,constants,nmax=n,progress=False)
        new_mat,eigenvals,eigenvecs = diagonalize(hamiltonian)
 
        #populate energy array
        for i,m in enumerate(levels):
            energies[i,j] = eigenvals[m]
     
    #check for convergence of highest energy level
    convergence=0
    for i in range(2,len(energies[0])):
        if abs(energies[-1,i]-energies[-1,i-2]) <= tolerance and convergence==0:
            convergence=dimensions[i-2]
        
    return energies, dimensions, levels, convergence

###################################################################################################

def plot_convergence_test(x,v_func,wave_func,nmin,nmax,constants,tolerance=1e-3,test_range=5):
    """Plots results from convergence function. Identical parameters and returns
    
    Parameters:
        x (array): set of points over which wave function is calculated
        v_func (function): potential energy function
        wave_func (function): stationary state wave function
        nmin, nmax (int): minimum and maximum energy levels to analyze. nmax must be greater than or equal to nmin
        tolerance (float): maximum difference between calculations for convergence to be true
        test_range (int): minimum number of tests to perform. Defaults to 5
    
    Returns:
        energies (2D array): energy of each level at each step
        dimensions (1D array): rows (or columns) of Hamiltonian array at each step
        levels (1D array): energy levels being analyzed
        convergence (int): dimension at which highest analyzed energy converged. If 0, did not converge
    """

    #perform convergence test
    energies, dimensions, levels, convergence = convergence_test(x,v_func,wave_func,nmin,nmax,constants,tolerance=tolerance,test_range=test_range)
    
    #plot results of convergence test
    for ind,row in enumerate(energies):
        plt.plot(dimensions,row,label="n="+str(levels[ind]))
    
    return energies, dimensions, levels, convergence

