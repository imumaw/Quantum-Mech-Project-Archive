#Isaiah Mumaw
#test code for Project 2

#general
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#main code
import project2 as main

##############################################################################
#SIMULATIONS LIST
##############################################################################

#set value True for each simulation being run. Full simulations below

#for the love of God, do not try to run in Spyder <3
#diatomic simulations require the correct file pathway, make sure to change that if you want to run them

run_hydrogen = False
run_helium = False
run_dihydrogen = False
run_dideuterium = False

##############################################################################
#MONOATOMIC HYDROGEN SIMULATION
##############################################################################

if run_hydrogen:
    
    print(" ")
    print("RUNNING HYDROGEN SIMULATION")
    print(" ")
    
    #set up relevant variables
    a_array = np.arange(0.5,1.5,0.01)
    a_len = len(a_array)
    hydrogen_energies = np.zeros(a_len)
    acc_rates = np.zeros(a_len)
    
    print("Getting energies...")
    print(20*"-"+"  0.0%", end="\r")
    
    for i,a in enumerate(a_array):
        energy,acc = main.metropolis_int(main.H_mono, main.hydrogen, dim=3, size=100000, scale=0.55, acceptance=True, a=a, z=1)
        hydrogen_energies[i] = energy
        acc_rates[i] = acc
        
        bar = math.floor(20*(i/a_len))
        print(bar*"█"+(20-bar)*"-"+"  "+str(round(100*i/a_len,1))+"%", end="\r")
        
    print(20*"█"+"  100.0%")

    print("Ground state energy is: "+str(min(hydrogen_energies)))
    print("Average acceptance rate: "+str(np.mean(acc_rates)))
    
    print(" ")
    
    print("Generating plot...")
    plt.figure("Hydrogen Plot 1/1")
    main.energy_curve(a_array,hydrogen_energies,title="Hydrogen energies vs. minimization constant")
    print("Plot generated")
    plt.show()
    
    print(" ")
    print("HYDROGEN SIMULATION COMPLETED")
    print(" ")
    
##############################################################################
#MONOATOMIC HELIUM SIMULATION
##############################################################################

if run_helium:
    
    print(" ")
    print("RUNNING HELIUM SIMULATION")
    print(" ")
    
    #set up relevant variables
    a_array = np.arange(0.3,1.3,0.01)
    a_len = len(a_array)
    helium_energies = np.zeros(a_len)
    acc_rates = np.zeros(a_len)
    
    print("Getting energies...")
    print(20*"-"+"  0.0%", end="\r")
    
    for i,a in enumerate(a_array):
        energy,acc = main.metropolis_int(main.H_mono, main.helium, dim=6, size=100000, scale=0.3, acceptance=True, a=a, z=2)
        helium_energies[i] = energy
        acc_rates[i] = acc
        
        bar = math.floor(20*(i/a_len))
        print(bar*"█"+(20-bar)*"-"+"  "+str(round(100*i/a_len,1))+"%", end="\r")
        
    print(20*"█"+"  100.0%")

    print("Ground state energy is: "+str(min(helium_energies)))
    print("Average acceptance rate: "+str(np.mean(acc_rates)))
    
    print(" ")
    
    print("Generating plot...")
    plt.figure("Helium Plot 1/1")
    main.energy_curve(a_array,helium_energies,title="Helium energies vs. minimization constant")
    print("Plot generated")
    plt.show()
    
    print(" ")
    print("HELIUM SIMULATION COMPLETED")
    print(" ")

##############################################################################
#DIHYDROGEN SIMULATION
##############################################################################

if run_dihydrogen:
    
    print(" ")
    print("RUNNING DIHYDROGEN SIMULATION")
    print(" ")
    
    #set up relevant variables
    a_array = np.arange(0.6,1.1,0.025)
    a_len = len(a_array)
    
    R_array = np.arange(1.0,1.7,0.025)
    R_len = len(R_array)
    
    dihydrogen_energies = np.zeros((R_len,a_len))
    acc_rates = np.zeros((R_len,a_len))
    
    print("Getting energies...")
    print(20*"-"+"  0.0%", end="\r")

    for i,R in enumerate(R_array):
        for j,a in enumerate(a_array):
            energy,acc = main.metropolis_int(main.H_dihydrogen, main.dihydrogen, dim=6, size=10000, scale=0.75, acceptance=True, a=a, R=R, z=1)
            dihydrogen_energies[i][j] = energy
            
            acc_rates[i][j] = acc
        
            bar = math.floor(20*(((i*a_len)+j)/(a_len*R_len)))
            print(bar*"█"+(20-bar)*"-"+"  "+str(round(100*((i*a_len)+j)/(a_len*R_len),1))+"%", end="\r")
    
    print(20*"█"+"  100.0%")

    print("Ground state energy is: "+str(np.min(dihydrogen_energies)))
    print("Average acceptance rate: "+str(np.mean(acc_rates)))
    
    #save file for use in dideuterium case
    np.savetxt("/Users/isaiahmumaw/Desktop/energy.txt",dihydrogen_energies)
    
    #get minimized E with respect to R
    min_index = np.argwhere(dihydrogen_energies==np.min(dihydrogen_energies))
    column = min_index[0][1]
    E_R = dihydrogen_energies[:,column]
    
    #now get data
    T_rot, E_rotational = main.T_rotational(R_array,E_R)
    T_vib, w, E_harmonic = main.T_vibrational(R_array,E_R)
    
    print("Rotational energies begin at T="+str(T_rot))
    print("Vibrational energies begin at T="+str(T_vib)+", with angular frequency of w="+str(w))
    
    print(" ")
    
    print("Generating plots...")
    
    plt.figure("Dihydrogen Plot 1/2")
    main.energy_heatmap(a_array,R_array,dihydrogen_energies,title="Dihydrogen energies vs. minimization constant")
    
    plt.figure("Dihydrogen Plot 2/2")
    ground_index = np.argmin(E_R)
    norm_R = R_array-R_array[ground_index]
    norm_E = E_R-E_R[ground_index]
    plt.plot(norm_R,norm_E,label="Exact harmonic energies")
    plt.plot(norm_R, 0.5*1836*w**2*norm_R**2, label="Approximated harmonic energies")
    plt.legend()
    plt.title("Generated fit line using w≈"+str(round(w,5)))
    
    print("Plots generated")
    plt.show()
    
    print(" ")
    print("DIHYDROGEN SIMULATION COMPLETED")
    print(" ")


##############################################################################
#DIDEUTERIUM SIMULATION
##############################################################################

if run_dideuterium:
    
    print(" ")
    print("RUNNING DIDEUTERIUM SIMULATION")
    print(" ")
    
    #set up relevant variables
    a_array = np.arange(0.6,1.1,0.025)
    a_len = len(a_array)
    
    R_array = np.arange(1.0,1.7,0.025)
    R_len = len(R_array)
    
    print("Loading energies from dihydrogen simulation (make sure file is properly indexed!)")
    dideuterium_energies = np.loadtxt("/Users/isaiahmumaw/Desktop/energy v2.txt")
    
    
    print("Ground state energy is: "+str(np.min(dideuterium_energies)))
    
    #get minimized E with respect to R
    min_index = np.argwhere(dideuterium_energies==np.min(dideuterium_energies))
    column = min_index[0][1]
    E_R = dideuterium_energies[:,column]
    
    #now get data
    T_rot, E_rotational = main.T_rotational(R_array,E_R,m=3675)
    T_vib, w, E_harmonic = main.T_vibrational(R_array,E_R,m=3675)
    
    print("Rotational energies begin at T="+str(T_rot))
    print("Vibrational energies begin at T="+str(T_vib)+", with angular frequency of w="+str(w))
    
    print(" ")
    
    print("Generating plots...")
    
    plt.figure("Dihydrogen Plot 1/2")
    main.energy_heatmap(a_array,R_array,dideuterium_energies,power=True,title="Dihydrogen energies vs. minimization constant")
    
    plt.figure("Dihydrogen Plot 2/2")
    ground_index = np.argmin(E_R)
    norm_R = R_array-R_array[ground_index]
    norm_E = E_R-E_R[ground_index]
    plt.plot(norm_R,norm_E,label="Exact harmonic energies")
    plt.plot(norm_R, 0.5*3675*w**2*norm_R**2, label="Approximated harmonic energies")
    plt.legend()
    plt.title("Generated fit line using w≈"+str(round(w,5)))
    
    print("Plots generated")
    plt.show()
    
    print(" ")
    print("DIDEUTERIUM SIMULATION COMPLETED")
    print(" ")
