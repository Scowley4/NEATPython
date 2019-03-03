import numpy as np
from flappyNeat import main
import sys

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




def fit_pole_balance(network,
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

    Returns
    -------
    fitness (float) total seconds the pole was upright
    """

    #Differential equations (I've checked these by hand):

    def angular_accel(theta,theta_vel,force):

        # This equation comes from the paper cited above

        a = grav*np.sin(theta)+np.cos(theta)
        b = (-1*force - pole_mass*pole_len*(theta_vel**2)*np.sin(theta))/(cart_mass+pole_mass)
        c = pole_len*(4./3 - ((pole_mass*(np.cos(theta))**2)/cart_mass+pole_mass))
        theta_acc = a*b/float(c)

        return theta_acc

    def cart_accel(theta,theta_vel,theta_acc,force):

        # This equation comes from the paper above
        a = (theta_vel**2)*np.sin(theta) - theta_acc*np.cos(theta)
        x_acc = (force + pole_mass*pole_len*a)/(cart_mass+pole_mass)

        return x_acc


    # Initial conditions: all zero
    theta = 0
    theta_vel = 0
    theta_acc = 0
    x = 0
    x_vel = 0
    x_acc = 0
    tau = time_step

    #Run simulation
    time_upright = 0

    while (abs(x) < track_limit) and (abs(theta) < pole_angle_failure):
        #Get the force from the network
        force = network.activate([theta,theta_vel,x,x_vel])

        #Compute the system parameters for the next time step
        next_theta = theta + tau*theta_vel
        next_theta_vel = theta_vel + tau*theta_acc
        next_theta_acc = angular_accel(next_theta,next_theta_vel,force)

        next_x = x + tau*x_vel
        next_x_vel = x_vel + tau*x_acc
        next_x_acc = cart_accel(next_theta,next_theta_vel,next_theta_acc,force)

        time_upright += tau

    return time_upright


def fit_flappy(network):
    "Returns time the bird survived"
    fitness = main(network)
    return fitness
