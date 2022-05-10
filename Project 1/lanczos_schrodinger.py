"""Isaiah Mumaw and Dan Palmer
University of Notre Dame


HOW TO USE (much easier than the other module):
    
Get stationary states:
1) Run generate_mat with appropriate arguments included
2) Run eigen with the resulting matrix over the same range of points
    -will return eigenvals and eigenvecs
    
Test for convergence:
    -test_range_convergence() will test the convergence when adjusting the
     magnitude of x
    -test_step_convergence() will test the convergence when adjusting the
     step size
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint

###################################################################################################

def generate_mat(xmin, xmax, H, potential_func, constants):
    """Calculates wave function algorithmically
    
    Parameters:
        xmin, xmax (numbers): starting and ending points along x-axis
        H (number): step size
        potential_func (function): potential energy function
        constants (dictionary): any additional constants needed
    """
    
    E0= -1/H**2
    
    #set up initial arrays
    x=np.arange(xmin,xmax,H)
    num_steps=len(x)
    mat=np.zeros((num_steps,num_steps))

    #find potential at all points
    potential=potential_func(x,constants)
    
    #setting up corners of array
    D_array=get_d(potential,E0)
    mat[0,0]=D_array[0]
    mat[0,1]=E0
    mat[num_steps-1,num_steps-1]=D_array[-1]
    mat[num_steps-1,num_steps-2]=E0
    
    #calculate diagonal and diagonal-adjacent elements
    for i in range(1,num_steps-1):
        mat[i,i-1]=E0
        mat[i,i]=D_array[i]
        mat[i,i+1]=E0
    
    return mat

###################################################################################################

def get_d(pot_val,E0):
    """finds D value given potential
    
    Parameters:
        pot_val (array-like): potential energy value(s)
        E0 (number): value of E0 given by -1/H**2
    """
    return (2*(-E0+pot_val))

###################################################################################################

def eigen(mat,xmin,xmax,H):
    """gets eigenvalues/eigenvectors, sorted by energy level
    
    Parameters:
        mat (array): matrix given by generate_mat()
        xmin, xmax (numbers): starting and ending points along x-axis
        H (number): step size
    """
    x=np.arange(xmin,xmax,H)

    eig_val,eig_vect=np.linalg.eigh(mat) 
    norm_const = np.sqrt(1/spint.simps((eig_vect[:,1])**2,x))   #normalize
    
    return eig_val/2,eig_vect*norm_const

###################################################################################################

def test_range_convergence(rmin,rmax,H,n,potential_func,constants,tol=1e-6):
    """Tests for convergence across a range of x values. Returns x and y axis arrays
    
    Parameters:
        rmin, rmax (numbers): starting maximum/final maximum values on x-axis (0 < rmin < rmax).
        H (number): step size
        n (non-negative integer): energy level
        potential_func (function): potential energy function
        constants (dictionary): any additional constants needed
         tol (number, optional): tolerance needed for convergence to be true (defaults 1e-3)
    """
    
    x_convergence=False
    converges=False
    
    #set up arrays
    r_vals=np.arange(rmin,rmax,H)
    energy_vals=np.zeros_like(r_vals)
    
    #cycle through list of all maximum r_vals
    for i,x in enumerate(r_vals):
        mat=generate_mat(-x,x,H,potential_func,constants)
        eig_val,eig_vect=eigen(mat,-x,x,H)
        
        #get relevant energy value
        energy_vals[i]=eig_val[n]
        
        #check for convergence
        if abs(energy_vals[i]-energy_vals[i-1]) < tol and converges==False:
            x_convergence = x
            converges=True
    
    return r_vals,energy_vals,x_convergence

###################################################################################################

def test_step_convergence(hmin,hmax,r,n,potential,constants,tol=1e-3):
    """Tests for convergence by varying step size
    
    Parameters:
        hmin, hmax (numbers): min and max step sizes (0<hmin<hmax).
        r (number): maximum x magnitude
        n (non-negative integer): energy level
        potential_func (function): potential energy function
        constants (dictionary): any additional constants needed
        tol (number, optional): tolerance needed for convergence to be true (defaults 1e-3)
    """

    h_convergence=False

    #set up arrays
    h_vals=np.arange(hmin,hmax,0.01)
    energy_vals=np.zeros_like(h_vals)
    
    #cycle through all h_vals
    for i,h in enumerate(h_vals):
        mat=generate_mat(-r,r,h,potential,constants)
        eig_val,eig_vect=eigen(mat,-r,r,h)
        
        #get relevant energy value
        energy_vals[i]=eig_val[n]
        
        #check for convergence
        if abs(energy_vals[i]-energy_vals[i-1]) < tol:
            h_convergence = h

    return h_vals,energy_vals,h_convergence
    