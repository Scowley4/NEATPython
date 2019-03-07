import numpy as np
import sys
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def fit_pole_balance(network,
                     max_time=120,
                     grav=-9.81,
                     pole_len=.5,
                     track_limit=2.4,
                     pole_angle_failure=np.pi*.5,
                     cart_mass=1.,
                     pole_mass=.1,
                     time_step=.02,
                     display=False):

    """
    Pole balancing problem from:
    https://researchbank.swinburne.edu.au/file
    /62a8df69-4a2c-407f-8040-5ac533fc2787/1
    /PDF%20%2812%20pages%29.pdf

    Parameters
    ----------
    network (neat network object) that accepts two inputs and has one output
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
    pole_loc = []


    while (abs(x) < track_limit) and (abs(theta) < pole_angle_failure) and (t[0] < max_time):
        y0 = np.array([x,x_vel,theta,theta_vel])

        # Network chooses the direction to push
        push = network.activate([x,theta])

        if push > .5:
            force = 1.
        else:
            force = -1.
        #force = push*10

        F = makeSystem(force)
        # Simulate system
        x,x_vel,theta,theta_vel = odeint(F,y0,t)[1]
        pole_loc.append((x,theta,t[0]))
        t = np.linspace(t[1],t[1]+time_step,2)

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-2, 2), ylim=(-2, 2))
        pole, = ax.plot([], [], '-', color='#70460f',lw=4)
        cart, = ax.plot([], [], '-', color='k',lw=8)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        def init():
            """initialize animation"""
            pole.set_data([], [])
            cart.set_data([], [])
            time_text.set_text('')
            return pole,cart,time_text
        
        def animate(i):
            x,theta,t = pole_loc[i]
            pole.set_data(np.array([x,x+2*pole_len*np.sin(theta)]),
                      np.array([0,2*pole_len*np.cos(theta)]))
            cart.set_data(np.array([x-.2,x+.2]),np.array([0,0]))
            time_text.set_text("Time: {}".format(t))
        
        ani = animation.FuncAnimation(fig, animate, frames=len(pole_loc),
                                      init_func=init)  
        plt.show()
        #animation.writer = animation.writers['ffmpeg'] 
        #ani.save("poleBal.mp4",writer=writer)
        plt.close()
        
    return t[0]
