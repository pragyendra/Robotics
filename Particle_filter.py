"""
COMS 4733 Fall 2021 Homework 4
Scaffolding code for localization using a particle filter
Inspired by a similar example on the PythonRobotics project
https://pythonrobotics.readthedocs.io/en/latest/
"""

import math
import matplotlib.pyplot as plt
import numpy as np


# "True" robot noise (filters do NOT know these)
WHEEL1_NOISE = 0.05
WHEEL2_NOISE = 0.1
BEARING_SENSOR_NOISE = np.deg2rad(1.0)

# Physical robot parameters (filters do know these)
RHO = 1
L = 1
MAX_RANGE = 20.0    # maximum observation range

# RFID positions [x, y]
RFID = np.array([[-5.0, -5.0],
                 [10.0, 0.0],
                 [10.0, 10.0],
                 [0.0, 15.0],
                 [-5.0, 20.0]])

# Covariances used by the estimators
Q = np.diag([0.1, 0.1, np.deg2rad(1.0)]) ** 2
R = np.diag([0.4, np.deg2rad(1.0)]) ** 2

# Other parameters
DT = 0.1            # time interval [s]
SIM_TIME = 30.0     # simulation time [s]
NP = 100            # Number of particles

# Plot limits
XLIM = [-20,20]
YLIM = [-10,30]
show_animation = True


"""
Robot physics
"""
def input(time, x):
    # Control inputs to the robot at a given time for a given state
    psi1dot = 3.7
    psi2dot = 4.0
    return np.array([psi1dot, psi2dot])

def move(x, u):
    # Physical motion model of the robot: x_k = f(x_{k-1}, u_k)
    # Incorporates imperfections in the wheels
    theta = x[2]
    psi1dot = u[0] * (1 + np.random.rand() * WHEEL1_NOISE)
    psi2dot = u[1] * (1 + np.random.rand() * WHEEL2_NOISE)

    velocity = np.array([RHO/2 * np.cos(theta) * (psi1dot+psi2dot),
                         RHO/2 * np.sin(theta) * (psi1dot+psi2dot),
                         RHO/L * (psi2dot - psi1dot)])

    return x + DT * velocity

def measure(x):
    # Physical measurement model of the robot: z_k = h(x_k)
    # Incorporates imperfections in both range and bearing sensors
    z = np.zeros((0, 3))

    for i in range(len(RFID[:, 0])):
        dx = RFID[i, 0] - x[0]
        dy = RFID[i, 1] - x[1]
        r = math.sqrt(dx**2 + dy**2)
        phi = math.atan2(dy, dx) - x[2]

        if r <= MAX_RANGE:
            zi = np.array([[np.round(r),
                            phi + np.random.randn() * BEARING_SENSOR_NOISE,
                            i]])
            z = np.vstack((z, zi))

    return z


"""
Particle filtering procedures
"""
def localization(px, pw, z, u):
    # Particle filter procedures: Predict, update, resample when necessary
    for ip in range(NP):
        px[:,ip] = predict(px[:,ip], u)
        pw[0,ip] *= update(px[:,ip], z)

    if pw.sum() == 0:
        px, pw = generate_particles()

    pw = pw / pw.sum()
    x_est = px.dot(pw.T).flatten()

    N_eff = 1.0 / (pw.dot(pw.T))[0,0]
    if N_eff < NP/1.0:
        px = resample(px, pw)
        pw = np.ones((1, NP)) / NP

    return x_est, px, pw


def generate_particles():
    # Generate a set of NP particles with uniform weights

    p_x = np.random.uniform(XLIM[0], XLIM[1], NP)
    p_y = np.random.uniform(YLIM[0], YLIM[1], NP)
    p_th = np.random.uniform(-np.pi, np.pi, NP)

    px = np.vstack((p_x, p_y, p_th))
    pw = np.ones((1, NP)) / NP
    return px, pw


def predict(x, u):
    """
    :param x: Particle state (x,y,theta) [size 3 array]
    :param u: Robot inputs (u1,u2) [size 2 array]
    :return: Particle's updated state sampled from the motion model
    """

    w = np.random.multivariate_normal(np.zeros(3), Q)
    # print("Shape wwwwwwwww", np.shape(w))
    x[0] = x[0]+ DT*(RHO/2)*(np.cos(x[2]))*(u[0]+u[1]) + w[0]
    x[1] = x[1]+ DT*(RHO/2)*(np.sin(x[2]))*(u[0]+u[1]) + w[1]
    x[2] = x[2]+ DT*(RHO/L)*(u[1]-u[0]) + w[2]

    return x


def update(x, z):
    """
    :param x: Particle state (x,y,theta) [size 3 array]
    :param z: Sensor measurements [px3 array]. Each row contains range, bearing, and landmark's true (x,y) location.
    :return: Particle's updated weight
    """

    w = 1
    for i in range(len(z)):
      xl = RFID[int(z[i,2]),0]
      yl = RFID[int(z[i,2]),1]

      #creating z matrix
      z_n = np.zeros(2).reshape(2,1)
      z_n[0,0] = z[i,0]
      z_n[1,0] = z[i,1]


      #creating h matrix
      vk = np.random.multivariate_normal(np.zeros(2), R)
      h_ik = np.zeros(2).reshape(2,1)
      r_ik = (np.sqrt(np.square(xl -x[0])+np.square(yl - x[1])))
      h_ik[0,0] = r_ik + vk[0]
      h_ik[1,0] = np.arctan2(yl -x[1],xl -x[0]) - x[2] + vk[1]

      y_i = z_n - h_ik

      if r_ik > MAX_RANGE:
        return 0
      else:
        w = w *((1/(2*np.pi*(np.sqrt(np.linalg.det(R))))) * np.exp((-1/2*y_i.T) @ np.linalg.inv(R) @ y_i) )

      # print("wwwwww", w)
    return float(w)


def resample(px, pw):
  #hohohey
    """
    :param px: All current particles [3xNP array]
    :param pw: All particle weight [size NP array]
    :return: A new set of particles obtained by sampling the original particles with given weights
    """

    w_cum = np.zeros(len(pw[0]))
    for i in range(len(pw[0])):
      w_cum[i] = (w_cum[i-1] + pw[0,i])

    # print("w_cum", w_cum)

    sample = 10
    new_px = np.zeros_like(px)
    done = 0
    while done < sample:
      # print("total", w_cum)
      num_rand = np.random.uniform(0, w_cum[-1])
      # print("num_rand", num_rand)
      for i in range(len(w_cum)):
        if num_rand <= w_cum[i]:
          new_px[:,done] = px[:,i]
          done +=1
          break
    # print("new_px pppppppppppppppppppppppppppppppppp")
    # print(new_px[:,2])
    return new_px


def main():
    time = 0.0

    # Initialize state
    x_est = np.zeros(3)
    x_true = np.zeros(3)

    # State history
    h_x_est = x_est.T
    h_x_true = x_true.T

    px, pw = generate_particles()

    while time <= SIM_TIME:
        time += DT
        u = input(time, x_true)
        x_true = move(x_true, u)
        z = measure(x_true)
        x_est, px, pw = localization(px, pw, z, u)

        # store data history
        h_x_est = np.vstack((h_x_est, x_est))
        h_x_true = np.vstack((h_x_true, x_true))

        if show_animation:
            plt.cla()

            for i in range(len(z[:,0])):
                plt.plot([x_true[0], RFID[int(z[i,2]),0]], [x_true[1], RFID[int(z[i,2]),1]], "-k")
            plt.plot(RFID[:,0], RFID[:,1], "*k")
            plt.plot(px[0,:], px[1,:], ".g")
            plt.plot(np.array(h_x_true[:,0]).flatten(),
                     np.array(h_x_true[:,1]).flatten(), "-b")
            plt.plot(np.array(h_x_est[:,0]).flatten(),
                     np.array(h_x_est[:,1]).flatten(), "-r")

            plt.axis("equal")
            plt.xlim(XLIM)
            plt.ylim(YLIM)
            plt.grid(True)
            plt.pause(0.001)

    plt.figure()
    errors = np.abs(h_x_true - h_x_est)
    plt.plot(errors)
    dth = errors[:,2] % (2*np.pi)
    errors[:,2] = np.amin(np.array([2*np.pi-dth, dth]), axis=0)
    plt.legend(['x error', 'y error', 'th error'])
    plt.xlabel('time')
    plt.ylabel('error magnitude')
    plt.ylim([0,1.5])
    plt.show()


if __name__ == '__main__':
    main()
