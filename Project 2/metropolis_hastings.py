import numpy as np
import math
import matplotlib.pyplot as plt

##############################################################################

def metropolis_int(h, dist_func, dim, size=100000, scale=1, progress=False, **kwargs):
    """Generates points based on Metropolis-Hastings algorithm
    
    Parameters:
        h: function to integrate divided by distribution function
        dist_func (func): distribution function p(x)
        size (int, optional): number of points to generate (defaults 100000)
        progress (bool, optional): if True, displays progress bar
        **kwargs
    """
    
    rand_vals = np.random.rand(size)   #between 0 and 1, compare to p_move
    rand_chain = np.random.normal(loc=0,scale=scale,size=(dim,size)) #generate steps
    
    mean = 0
    
    naccept = 0
    
    for j in range(1,size):

        #get current and proposed coordinates
        r1 = rand_chain[:,j-1]
        r2 = r1 + rand_chain[:,j]
        
        #get probabilities at each coordinate
        p1 = dist_func(r1,**kwargs)
        p2 = dist_func(r2,**kwargs)
        
        #probability of moving
        p_move = p2 / p1
        
        #determine if move happens
        if rand_vals[j] < p_move:
            rand_chain[:,j]=r2
            naccept+=1
        else:
            rand_chain[:,j]=r1
        
        mean += h(rand_chain[:,j],**kwargs)
        
        #progress bar
        if progress and j%10000==0:
            bar = math.floor(40*(j/size))
            print(bar*"█"+(40-bar)*"-"+"  "+str(round(100*j/size,1))+"%", end="\r")
     
    #full progress bar
    if progress:        
        print(40*"█"+" 100.0%")
    
    acceptrate = naccept/size
    print (f"Acceptance rate is: {acceptrate}")
    #Calculate the integral
    return mean/size

##############################################################################

def convergence_test(f,dist_func,dim,size_min,size_max,steps=30,**kwargs):
    """
    """
    
    sizes = np.linspace(size_min,size_max,steps,dtype=int)
    int_array = np.empty(steps)
    
    for i,size in enumerate(sizes):
        integral = metropolis_int(f,dist_func,dim,size=size,**kwargs)
        int_array[i] = integral
    
    return int_array, sizes
    
    