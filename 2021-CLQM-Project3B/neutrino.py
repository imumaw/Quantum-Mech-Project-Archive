#calculates neutrino oscillations through matter
#Isaiah Mumaw, Andrew Langford, Tianyi Wang

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

##############################################################################
##############################################################################

#Global constants
theta = np.radians(33.6)                        #experimental value of theta for U_pmns
dm2c4 = 7.53*10**(-5)                           #delta_m^2*c^4 in units of eV^2
h_bar = 6.58211951*10**-16                      #hbar in eV*s       
c = 3e10                                        #speed of light in units of cm/s
Gf = 1.1663787e-5 / (1e9 * 1e9) * (h_bar*c)**3  #Fermi constant in units of eV*cm^3
y_e = 0.5                                       #electron ratio
m_p = 1.6726219e-24                             #proton mass in units of g

##############################################################################
##############################################################################

def adjoint(M):
    #gets the hermitian conjugate of a matrix
    return np.conj(np.transpose(M))

def hamiltonian(rho, E):
    """ hamiltonian -- given an inital state in flavor basis and medium density,
            the function returns the Hamiltonian eigenstates and eigenvalues

            Arguments:
                rho (double): propogation medium density in g/cm^3
                E (double): energy of the initial state in eV

            Returns:
                eigvals (array): array of Hamiltonian eigenvalues
                U_msw (matrix): matrix of Hamiltonian eigenstates as column vectors
    """

    #transformation matrix from mass to flavor basis
    U_pmns = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])

    #Kinetic and potential terms
    H_mass = np.array([[0,0],[0,dm2c4/(2*E)]])              #mass basis
    V = np.array([[np.sqrt(2)*Gf*y_e*rho/m_p,0],[0,0]])     #flavor basis

    #full Hamiltonian
    H_msw = U_pmns @ H_mass @ adjoint(U_pmns) + V
    eigvals, U_msw = np.linalg.eigh(H_msw)

    return eigvals, U_msw


def p_evolve(t, nu_0, rho, E, nu_e0=[1,0]):
    """ time_evolve - function calculates the probability of an initial flavor basis
            state, nu_e0, being in its initial state as a function of time

                Arguments:
                    t (array): time array
                    nu_e0 (2x1 array): initial flavor basis state to propogate
                    rho (function): propogation medium density in g/cm^3
                    E (double): energy of the initial state in eV
                    nu_e0 (2x1 array, optional): comparison state for probabilities (defaults to be fully in electron neutrino state)

                Returns:
                    p (array): probability array
    """

    #set up variables
    dt = t[1]-t[0]
    rho_array = rho(t)
    
    p_array = np.zeros(len(t))

    #initial state
    nu_e = nu_0

    for i in range(len(t)):
        
        # obtain eigenvals and transformation matrix
        lamda, U_msw  = hamiltonian(rho_array[i], E)
    
        # Time evolution matrix
        tev_mat = np.array([[np.exp(-1j*lamda[0]*dt/h_bar), 0], [0, np.exp(-1j*lamda[1]*dt/h_bar)]])
    
        #initial state in matter basis
        nu_msq_0 = adjoint(U_msw) @ nu_e
    
        #time propagate in matter basis
        nu_msq_t = tev_mat @ nu_msq_0
        
        #convert back to flavor basis
        nu_e = adjoint(U_msw) @ nu_msq_t
        
        #update probability
        p_array[i] = abs(adjoint(nu_e) @ nu_e0)**2

    return p_array, nu_e

##############################################################################
#simulations for report below!
##############################################################################

if __name__ == '__main__':

    ###########################################################################
    #neutrino oscillations through free space (vary energy)
    ###########################################################################
    """
    #set up variables
    E_array = np.array([1e5, 1e6, 1e7])
    nu0 = np.array([1,0])
    t = np.linspace(0, 0.001, 100000)
    
    def rho_space(t):
        #density function for free space
        d = t*c
        return 0 * d

    for E in E_array:
        p, nu_e = p_evolve(t, nu0, rho_space, E)
        plt.plot(t*c, p, label="E="+"{:.1e}".format(E)+" eV")
        
    plt.legend(loc=1)
    plt.title("Neutrino oscillation in free space")
    plt.ylabel("Probability of electron neutrino")
    plt.xlabel("Distance (cm)")
    """
    
    ###########################################################################
    #neutrino oscillations through constant density
    ###########################################################################
    """
    #set up variables
    E = 5e6
    rho_array = np.linspace(100,1000,6)
    nu0 = np.array([1,0])
    t = np.linspace(0, 0.001, 10000)
    
    def rho_constant(t):
        #density function for constant density (defined globally because of how I tested it)
        d = t*c
        return 0 * d + k
    
    #cycle through densities
    for rho_val in rho_array:
        k = rho_val
        p, nu_e = p_evolve(t, nu0, rho_constant, E)
        plt.plot(t*c, p, label="{:.1f}".format(rho_val)+" g/cm3")

    #plot it
    plt.legend(title="Density")
    plt.title("Neutrino oscillation in high constant density, E=5e+6 eV")
    plt.ylabel("Probability of electron neutrino")
    plt.xlabel("Distance (cm)")
    """
    
    ###########################################################################
    #solar neutrino emissions
    ###########################################################################
    
    #constants (still in CGS units)
    sun_density_max = 150
    earth_density = 5.5
    R_sun = 6.957e10
    R_core = 0.2*R_sun
    R_earth = 6.3e8
    R_orbit = 1.1496e12
    flux_core = 5e15
    
    nu0 = np.array([1,0])
    
    #density functions
    
    def rho_sun(t):
        #density function for Sun's density
        d = t*c
        return sun_density_max * np.exp(-d/(0.1*R_sun))
    
    def rho_space(t):
        #density function for free space
        d = t*c
        return 0 * d
    
    def rho_earth(t):
        #density function for Earth's density
        d = t*c
        return 0*d + earth_density
    
    ###########################################################################
    
    #first set of figures (change E_test to get both plots)
    """
    E_test = 2e7 #alt: 1e6
    
    #time in each region
    t_sun = np.linspace(0,R_sun,100000)/c
    t_space = np.linspace(R_sun,R_sun+R_orbit,1000000)/c
    t_earth = np.linspace(R_sun+R_orbit,R_sun+R_orbit+R_earth,100000)/c
    
    #time propagation
    p_sun, nu_e_sun = p_evolve(t_sun, nu0, rho_sun, E_test)
    p_space, nu_e_space = p_evolve(t_space, nu_e_sun, rho_space, E_test)
    p_earth, nu_e_earth = p_evolve(t_earth, nu_e_space, rho_earth, E_test)

    #set up plots
    fig,(ax1,ax2) = plt.subplots(2,1)

    #plot the sun-space region
    ax1.plot(t_sun*c,p_sun,color="tab:orange")
    ax1.plot(t_space[0:25000]*c,p_space[0:25000],color="tab:blue")
    
    ax1a = ax1.twinx()
    ax1a.plot(t_sun*c,rho_sun(t_sun),color="black")
    ax1a.plot(t_space[0:25000]*c,rho_space(t_space[0:25000]),color="black")

    ax1.set_xlabel("Distance")
    ax1.set_ylabel("Probability")
    ax1a.set_ylabel("Density")
    ax1.set_ylim(-0.1,1.1)
    ax1a.set_ylim(-5,155)

    #plot the space-earth region
    ax2.plot(t_space[998999:999999]*c,p_space[998999:999999],color="tab:blue")
    ax2.plot(t_earth*c,p_earth,color="tab:green")
    
    ax2a = ax2.twinx()
    ax2a.plot(t_space[998999:999999]*c,rho_space(t_space[998999:999999]),color="black")
    ax2a.plot(t_earth*c,rho_earth(t_earth),color="black")
    
    ax2.set_xlabel("Distance")
    ax2.set_ylabel("Probability")
    ax2a.set_ylabel("Density")
    ax2.set_ylim(-0.1,1.1)
    ax2a.set_ylim(-5,155)
    """

    #second figure
    """
    #energies
    E_array = np.logspace(5,7.30103,num=25,base=10.0)
    
    #probability arrays
    p_ave_nearside = np.zeros(len(E_array))
    p_ave_farside = np.zeros(len(E_array))
    p_max_nearside = np.zeros(len(E_array))
    p_max_farside = np.zeros(len(E_array))
    p_min_nearside = np.zeros(len(E_array))
    p_min_farside = np.zeros(len(E_array))
    
    #time arrays
    t_sun = np.linspace(0,R_sun,100000)/c
    t_space = np.linspace(R_sun,R_sun+R_orbit,100000)/c
    t_earth = np.linspace(R_sun+R_orbit,R_sun+R_orbit+R_earth,100000)/c
     
    #cycle through all energies
    for i,E in enumerate(E_array):

        #propagation
        p_sun, nu_e_sun = p_evolve(t_sun, nu0, rho_sun, E)
        p_space, nu_e_space = p_evolve(t_space, nu_e_sun, rho_space, E)
        p_earth, nu_e_earth = p_evolve(t_earth, nu_e_space, rho_earth, E)
        
        #probabilities
        p_ave_nearside[i] = np.mean(p_space)
        p_ave_farside[i] = np.mean(p_earth)
        p_max_nearside[i] = np.max(p_space)
        p_max_farside[i] = np.max(p_earth)
        p_min_nearside[i] = np.min(p_space)
        p_min_farside[i] = np.min(p_earth)
        
    #plot everything
    plt.plot(E_array,p_ave_nearside,label="Near side", color="tab:orange")
    plt.plot(E_array,p_ave_farside,label="Far side", color="tab:blue")
    
    plt.plot(E_array,p_max_nearside,label="Near side max", color="tab:orange", linestyle=":")
    plt.plot(E_array,p_max_farside,label="Far side max", color="tab:blue", linestyle=":")
    
    plt.plot(E_array,p_min_nearside,label="Near side min", color="tab:orange", linestyle="--")
    plt.plot(E_array,p_min_farside,label="Far side min", color="tab:blue", linestyle="--")
    
    #make it look nice
    plt.title("Probability of electron neutrinos on Earth")
    plt.legend(loc="upper left")
    plt.xlabel("Energy")
    plt.xscale("log")
    plt.ylabel("Probability")
    """
    
    #third set of figures
    """
    #energies
    E_array = np.logspace(5,7.30103,num=25,base=10.0)
    
    #probability arrays
    p_ave_nearside = np.zeros(len(E_array))
    p_ave_farside = np.zeros(len(E_array))
    p_ave_core = np.zeros(len(E_array))
    
    nu_e_current = flux_core * 4*np.pi*R_core
    
    #time arrays
    t_sun = np.linspace(0,R_sun,100000)/c
    t_space = np.linspace(R_sun,R_sun+R_orbit,100000)/c
    t_earth = np.linspace(R_sun+R_orbit,R_sun+R_orbit+R_earth,100000)/c
    
    #cycle through all energies
    for i,E in enumerate(E_array):
        
        #propagation
        p_sun, nu_e_sun = p_evolve(t_sun, nu0, rho_sun, E)
        p_space, nu_e_space = p_evolve(t_space, nu_e_sun, rho_space, E)
        p_earth, nu_e_earth = p_evolve(t_earth, nu_e_space, rho_earth, E)
        
        #get probabilities
        p_ave_core[i] = np.mean(p_sun[19500:20500])
        p_ave_nearside[i] = np.mean(p_space)
        p_ave_farside[i] = np.mean(p_earth)
    
    #calculate flux
    nu_current = nu_e_current / p_ave_core
    flux_earth_nearside = (nu_current / (4*np.pi*(R_sun+R_orbit)))
    flux_earth_farside = (nu_current / (4*np.pi*(R_sun+R_orbit+R_earth)))
    
    #plot it
    plt.plot(E_array,flux_earth_nearside,label="Total flux, nearside",color="tab:orange",linestyle="--")
    plt.plot(E_array,flux_earth_nearside*p_ave_nearside,label="Elecron neutrino flux, nearside",color="tab:orange")
    plt.plot(E_array,flux_earth_farside,label="Total flux, farside",color="tab:blue",linestyle=":")
    plt.plot(E_array,flux_earth_nearside*p_ave_farside,label="Electron flux, farside",color="tab:blue")
    
    #make it look nice
    plt.ylabel("Flux")
    plt.xlabel("Energy")
    plt.legend()
    plt.title("Solar neutrino flux at Earth")
    plt.xscale("log")
    """
    
    plt.show()
    