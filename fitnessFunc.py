import numpy as np

def fit_xor(network):
    """
    Function for determining how well the network solves the 
    exclusive or problem.
    
    Parameters
    ----------
    network (neat network object) that accepts two inputs and has one output
    
    Returns
    -------
    fitness (a fitness score)
    """
    samples = 4
    X = np.array([[0,0],[1,0],[0,1],[1,1]])
    Y = np.array([0,1,1,0])
    
    outputs = np.zeros(samples)
    for i in range(len(X)):
        outputs[i] = network.activate(X[i])
    
    fitness = np.sum(Y*(outputs>.5))/float(samples)
    
    return fitness

def fit_dparity(network,num_inputs,samples=100):
    """
    Function for determining how well the network solves the 
    d-parity problem. If x is a vector of zeros and ones
    dparity(x) returns 1 if there are an odd number of ones
    and returns 0 if there are an even number of ones.
    (Generalized xor)
    
    Parameters
    ----------
    network (neat network object) that accepts num_inputs and has one output
    
    Returns
    -------
    fitness (a fitness score)
    """
    
    # Create random vectors of zeros and ones
    X = 1*(np.random.normal(size=(samples,num_inputs)) > 0)
    # Compute parity of each row of X
    Y = ~(X.sum(axis=1)/2 == X.sum(axis=1)/2.)
    Y = Y*1
    
    outputs = np.zeros(samples)
    for i in range(samples):
        outputs[i] = network.activate(X[i])
        
    fitness = np.sum(Y*outputs)/float(samples)
    return fitness
        