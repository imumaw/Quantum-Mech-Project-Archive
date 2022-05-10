# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:23:47 2021

@author: Luca
"""

import numpy as np

def delta(m,n):
    '''

    Parameters
    ----------
    m : integer
    n : integer

    Returns
    -------
    integer
        A simple kroeneker delta (or however you spell the guy's name)

    '''
    if m != n:
        return 0
    else:
        return 1

def HO_energy(n, hbar, mass, omega):
    '''
    Parameters
    ----------
    n : integer
        nth energy level (ground state is n=0)
    hbar : float
        hbar, everybody knows hbar
    mass : float
        the mass of the oscillator
    omega : float
        the angular frequency of the oscillator

    Returns
    -------
    float
        Energy of the nth level of the harmonic oscillator
    '''
    
    return hbar*omega*(n + 0.5)

def m_Hprime_n( m, n, hbar, mass, omega):    
    '''
    Parameters
    ----------
    m : integer
        nth energy level (ground state is n=0)
    n : integer
        nth energy level (ground state is n=0)
    hbar : float
        hbar, everybody knows hbar
    mass : float
        the mass of the oscillator
    omega : float
        the angular frequency of the oscillator

    Returns
    -------
    float
        matrix element <m|H'|n> that you need to do perturbation theory
    '''

    '''
    the way this was derived is simply considering that H'=x^4, and then
    writing x in terms of raising and lowering operators. The delta functions
    come of course from terms like <m-2|n> and so forth. Notice that there are 
    16 terms since QM is a diva and apparently it's not considered classy enough
    to be commutative. It's for peasants like Newton and Galileo, not for pros like Plank
    '''
    m_Hprime_n = 0.25*( hbar/ (mass*omega) )**2 * \
      ( np.sqrt((n+1)*(n+2)*(n+3)*(n+4)) * delta(m,n+4) + \
        np.sqrt((n)*(n-1)*(n-1)*(n)) * delta(m,n) + \
        np.sqrt((n)*(n)*(n+1)*(n+2)) * delta(m,n+2) + \
        np.sqrt((n+1)*(n+1)*(n+1)*(n+2)) * delta(m,n+2) + \
        np.sqrt((n+1)*(n+2)*(n+2)*(n+1)) * delta(m,n) + \
        np.sqrt((n)*(n-1)*(n-2)*(n-3)) * delta(m,n-4) + 
        np.sqrt((n)*(n)*(n)*(n-1)) * delta(m,n-2) + \
        np.sqrt((n+1)*(n+1)*(n)*(n-1)) * delta(m,n-2) + \
        np.sqrt((n+1)*(n+2)*(n+2)*(n+2)) * delta(m,n+2) + \
        np.sqrt((n)*(n-1)*(n-2)*(n-1) * delta(m,n-2)) + \
        np.sqrt((n)*(n)*(n)*(n)) * delta(m,n) + \
        np.sqrt((n+1)*(n+1)*(n)*(n)) * delta(m,n) + \
        np.sqrt((n+1)*(n+2)*(n+3)*(n+3)) * delta(m,n+2) + \
        np.sqrt((n)*(n-1)*(n-1)*(n-1)) * delta(m,n-2) + \
        np.sqrt((n)*(n)*(n+1)*(n+1)) * delta(m,n) + \
        np.sqrt((n+1)*(n+1)*(n+1)*(n+1)) * delta(m,n) )
    
    return m_Hprime_n


def First_order_perturbation(n, hbar, mass, omega):
    '''
    Parameters
    ----------
    n : integer
        nth energy level (ground state is n=0)
    hbar : float
        hbar, everybody knows hbar
    mass : float
        the mass of the oscillator
    omega : float
        the angular frequency of the oscillator

    Returns
    -------
    float
        matrix element <n|H'|n> that is the first order correction to 
        the energy (notice that in principle the correction is eps*<n|H'|n>,
        but I include eps in the full expression, because I like it more like that,
        and you cannot stop me.)
    '''
    
    total_term = m_Hprime_n(n, n, hbar, mass, omega)
    
    return total_term

def Second_order_perturbation(n, hbar, mass, omega):
    '''
    Parameters
    ----------
    n : integer
        nth energy level (ground state is n=0)
    hbar : float
        hbar, everybody knows hbar
    mass : float
        the mass of the oscillator
    omega : float
        the angular frequency of the oscillator

    Returns
    -------
    float
        eq. 6.15 from Griffiths. Notice that the sum only goes from n-4 to n+4 since everything
        above and below is zero (Remember how in your matrix only certain terms are non-zero?
        Here it's the same thing. These terms will be killed by the deltas in m_Hprime_n)
    '''        
    total_term = 0
    for m in range(n-4,n+3):
        if m == n or m < 0:
            continue
        else:    
            total_term += np.abs(m_Hprime_n(m, n, hbar, mass, omega))**2 \
                / ( HO_energy(n, hbar, mass, omega) - HO_energy(m, hbar, mass, omega) )
    
    return total_term

def Energy_anharmonic_oscillator( eps, n, hbar=1, mass=1, omega=1 ):
    '''
    Parameters
    ----------
    eps : float
        magnitude of the perturbation Hamiltonian, assumed to be eps*x**4
    n : integer
        nth energy level (ground state is n=0)
    hbar (OPTIONAL): float
        hbar, everybody knows hbar
    mass (OPTIONAL): float
        the mass of the oscillator
    omega (OPTIONAL): float
        the angular frequency of the oscillator

    Returns
    -------
    float
        eq. 6.6 from Griffiths. 
    '''        
    
    return HO_energy(n, hbar, mass, omega) + \
            eps*First_order_perturbation(n, hbar, mass, omega) + \
            eps**2*Second_order_perturbation( n, hbar, mass, omega)
    
