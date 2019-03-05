import numpy as np
from flappyNeat import main
import sys
from scipy.integrate import odeint

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
        try:
            outputs[i] = network.activate(X[i])
        except:
            print(network.activate(X[i]))
            sys.exit()


    fitness = np.sum(Y*(outputs>.5))/float(samples)

    return fitness

def get_determined_fit_dparity(num_inputs):
    if num_inputs > 15:
        raise ValueError('Thats too big')
    samples = 2**num_inputs
    X = np.array([[int(x) for x in bin(i)[2:].zfill(num_inputs)]
                   for i in range(samples)])
    def determined_fit_dparity(network):
        # Compute parity of each row of X
        Y = X.sum(axis=1)%2

        outputs = np.zeros(samples)
        for i in range(samples):
            outputs[i] = network.activate(X[i])

        fitness = np.sum(Y*outputs)/float(samples)
        return fitness

    return determined_fit_dparity



def get_fit_dparity(num_inputs, samples=100):
    def fit_dparity(network):
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
        Y = X.sum(axis=1)%2

        outputs = np.zeros(samples)
        for i in range(samples):
            outputs[i] = network.activate(X[i])

        fitness = np.sum(Y*outputs)/float(samples)
        return fitness

    return fit_dparity



def fit_dparity(network, num_inputs, samples=100):
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
    Y = X.sum(axis=1)%2

    outputs = np.zeros(samples)
    for i in range(samples):
        outputs[i] = network.activate(X[i])

    fitness = np.sum(Y*outputs)/float(samples)
    return fitness

def fit_flappy(network):
    "Returns time the bird survived"
    fitness = main(network)
    return fitness

def fit_pole_balance(network,
                     max_time=120,
                     grav=-9.81,
                     pole_len=.5,
                     track_limit=2.4,
                     pole_angle_failure=.209,
                     cart_mass=1.,
                     pole_mass=.1,
                     time_step=.02):

    """
    Pole balancing problem from:
    https://researchbank.swinburne.edu.au/file
    /62a8df69-4a2c-407f-8040-5ac533fc2787/1
    /PDF%20%2812%20pages%29.pdf

    Parameters
    ----------
    network (neat network object) that accepts four inputs and has one output
    other parameters are explained in the paper
    max_time (int) the longest a network is permitted to keep the pole upright in seconds

    Returns
    -------
    fitness (float) total seconds the pole was upright
    """


    # Given a particular force value, create the system:
    
    def makeSystem(force):
        
        #Differential equations (I've checked these by hand):
        
        def angular_accel(theta,theta_vel):

            # This equation comes from the paper cited above

            a = grav*np.sin(theta)+np.cos(theta)
            b = (-1*force - pole_mass*pole_len*(theta_vel**2)*np.sin(theta))/(cart_mass+pole_mass)
            c = pole_len*(4./3 - ((pole_mass*(np.cos(theta))**2)/cart_mass+pole_mass))
            theta_acc = a*b/float(c)

            return theta_acc

        def cart_accel(theta,theta_vel,theta_acc):

            # This equation comes from the paper above
            
            a = (theta_vel**2)*np.sin(theta) - theta_acc*np.cos(theta)
            x_acc = (force + pole_mass*pole_len*a)/(cart_mass+pole_mass)

            return x_acc
        
        def F(x,t):
            theta_acc = angular_accel(x[2],x[3])
            x_acc = cart_accel(x[2],x[3],theta_acc)
            return np.array([x[1], x_acc, x[3], theta_acc])
        
        return F
        
    # Initial conditions:
    theta = 0
    theta_vel = 0
    theta_acc = 0
    x = 0
    x_vel = 0
    x_acc = 0
    t = np.linspace(0,time_step,2)


    while (abs(x) < track_limit) and (abs(theta) < pole_angle_failure) and (t[0] < max_time):
        y0 = np.array([x,x_vel,theta,theta_vel])
        force = network.activate([x,x_vel,theta,theta_vel])
        F = makeSystem(force)
        # Simulate system
        x,x_vel,theta,theta_vel = odeint(F,y0,t)[1]
        t = np.linspace(t[1],t[1]+time_step,2)
        
    return t[0]
        