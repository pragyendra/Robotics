# -*- coding: utf-8 -*-
"""problem6_Gradient.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RFcdMjDfGxqYw2NA1fKq74mcJIoYoQBN
"""

#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos

# Commented out IPython magic to ensure Python compatibility.
from mpl_toolkits import mplot3d
# %matplotlib inline

#creating a structure for our given Td Matrix
Td = np.zeros(16).reshape(4,4)

#check if the sturcture is correct
print(Td)

#manually assigning values for the 1st row as per given Td
Td[0,0] = 0.78
Td[0,1] = -0.494
Td[0,2] = 0.866
Td[0,3] = 1

#manually assigning values for 2nd row
Td[1,0] = 0.135
Td[1,1] = -0.855
Td[1,2] = -0.5
Td[1,3] = 2

#manually assigning values for 3rd row
Td[2,0] = 0.988
Td[2,1] = 0.156
Td[2,2] = 0
Td[2,3] = 2

#manually assigning values for 4th row
Td[2,0] = 0
Td[2,1] = 0
Td[2,2] = 0
Td[2,3] = 1

#checking if the Td values are correct
print(Td)

#defining the arm details global variables
global l1
global l2
global l3
l1 = 1
l2 = 1
l3 = 1

#defining a function to calculate J matrix
from numpy.core.fromnumeric import reshape
def give_J(x,y,z):
  l =1
  l1, l2, l3 = l, l, l
  theta1 = np.arctan2(y,x)
  theta3 = np.arccos(((x**2)+(y**2)+(z**2)-(2*l1*(np.sqrt((x**2)+(y**2))))+(l1**2)-(l2**2)-(l3**2))/(2*l2*l3))
  theta2 = np.arctan2(z,(np.sqrt((x**2)+(y**2))-l1))

  J = np.zeros(6*3).reshape(6,3)

  J[0,0] = (-sin(theta1)*l3*cos(theta2+theta3))-(sin(theta1)*l2*cos(theta2))-(l1*sin(theta1))
  J[0,1] = (-cos(theta1)*l3*sin(theta2+theta3))-(cos(theta1)*l2*sin(theta2))
  J[0,2] = (-cos(theta1)*l3*sin(theta2+theta3))

  J[1,0] = (cos(theta1)*l1*cos(theta2+theta3))+(cos(theta1)*l2*cos(theta2))+(l1*cos(theta1)) 
  J[1,1] = (-sin(theta1)*l3*sin(theta2+theta3))-(sin(theta1)*l2*sin(theta2))
  J[1,2] = (-sin(theta1)*l3*sin(theta2+theta3))

  J[2,0] = 0
  J[2,1] = (-sin(theta1)*l3*sin(theta2+theta3))
  J[2,2] = (l3*cos(theta2+theta3))

  J[3,0] = 0
  J[3,1] = sin(theta1)
  J[3,2] = sin(theta1)

  J[4,0] = 0
  J[4,1] = -cos(theta1)
  J[4,2] = -cos(theta1)

  J[5,0] = 1
  J[5,1] = 0
  J[5,2] = 0

  q = np.zeros(3).reshape(3,1)

  q[0,0] = theta1
  q[1,0] = theta2
  q[2,0] = theta3

  return J,q

#defining function that gives cordinates based on Td matrix
def give_cq(Td):
  cq = [Td[0,3],Td[1,3],Td[2,3]]
  return cq

cq2 = give_cq(Td)
#print the final cordinate configuration of the arm
print("cq2:", cq2)

# Defining Error Functoin
def error(q2,q1,Jq):
  diff = q2-q1
  err = Jq@diff
  return err

# defining a function to calculate T30 matrix as per given configuration in (theta1, theta2, theta3)
def give_Td(q):
  theta1 = q[0,0]
  theta2 = q[1,0]
  theta3 = q[2,0]

  Td = np.zeros(16).reshape(4,4)

  Td[0,0] = cos(theta1)*cos(theta2+theta3)
  Td[0,1] = -cos(theta2)*sin(theta2+theta3)
  Td[0,2] = sin(theta1)
  Td[0,3] = (cos(theta1)*l3*cos(theta2+theta3)) + (cos(theta1)*l2*cos(theta2)) + (l1*cos(theta1))

  Td[1,0] = sin(theta1)*cos(theta2+theta3)
  Td[1,1] = -sin(theta1)*sin(theta2+theta3)
  Td[1,2] = -cos(theta1)
  Td[1,3] = (sin(theta1)*l3*cos(theta2+theta3)) + (sin(theta1)*l2*cos(theta2)) + (l1*sin(theta1))

  Td[2,0] = sin(theta2 + theta3)
  Td[2,1] = cos(theta2 + theta3)
  Td[2,2] = 0
  Td[2,3] =  (l3*sin(theta2 + theta3)) + (l2*sin(theta2))

  Td[3,0] = 0
  Td[3,1] = 0
  Td[3,2] = 0
  Td[3,3] = 1

  return Td

#current configuration q = [0,0,0]
#l1 = l2 = l3 = l =1
q1 = np.zeros(3).reshape(3,1)

Tq1 = give_Td(q1)
cq1 = give_cq(Tq1)

#check if the cordinates we are getting is correct
print("cq1:",cq1)
print("-----------")

#calculating the Jq1 and q1 matrix for this given configuration
Jq1, q1 = give_J(cq1[0], cq1[1], cq1[2])

#printing our Jq1 matrix
print("Jq1: ", Jq1)
print("------------------")

#printing the q1 matrix
print("q1: ", q1)
print("-----------------")

Jq2, q2 = give_J(cq2[0], cq2[1], cq2[2])
#print the desired orientation of the arm
print("q2:", q2)

err = error(q2,q1,Jq1)
print("initial error:",err)
print("Norm of initial error: ",np.linalg.norm(err))

# defining Delta Theta
del_theta = np.zeros(3).reshape(3,1)
del_theta[0,0] = err[3,0]
del_theta[1,0] = err[4,0]
del_theta[2,0] = err[5,0]

# prinitng out the Delta theta
print("Delta theta:",del_theta)

# defining Delta O
del_o = np.zeros(3).reshape(3,1)
del_o[0,0] = err[0,0]
del_o[1,0] = err[1,0]
del_o[2,0] = err[2,0]

#printing Delta 0
print("Delta o:",del_o)

#cchecking the equations for Gradient Descent Algorithm
#tuning alpha hyperparameter
global alpha
alpha = 1

q_k1 = q1 + (alpha*Jq1.T@err)

print("qk1:",q_k1)

#initial cqk
cqk = cq1
print("cqk: ", cqk)

#defining the threshold epsilon
epsilon = 0.1

#for plotting error fx
error_plot = []

#plotting the arm
arm_x = []
arm_y = []
arm_z = []

#we already have a error matrix with norm greater than epsilon
print("norm before: ", np.linalg.norm(err))

steps = 1
#initiating the loop
while np.linalg.norm(err) > epsilon:
  # forward Kinematics
  Jk, qk = give_J(cqk[0], cqk[1], cqk[2])
  err = error(q2,qk,Jk)
  # err = Jk@np.round(q2-qk)
  qk = qk + (alpha*Jk.T@err)
  Tk = give_Td(qk)
  cqk = [Tk[0,3],Tk[1,3],Tk[2,3]]
  err = error(q2,qk,Jk)
  error_plot.append(np.linalg.norm(err))
  arm_x.append(qk[0,0])
  arm_y.append(qk[1,0])
  arm_z.append(qk[2,0])
  steps += 1

print("norm after:", np.linalg.norm(err))
print("Under thershold of", epsilon)
# print("In ", steps, "steps")

print("Final Angle Configuration we get is")
print(qk[0,0],qk[1,0],qk[2,0])
print("Desired Angle Configuration")
print(q2[0,0],q2[1,0],q2[2,0])
print("diff",q2-qk)
print("In ", steps, "steps")

print("error:", np.linalg.norm(err))

print("Final Cordinate Configuration we get is")
print(cqk[0],cqk[1],cqk[2])
print("Desired Cordinate Configuration")
print(cq2[0],cq2[1],cq2[2])

#plotting the error
plt.plot(error_plot)
plt.show()

ax = plt.axes(projection='3d')
# Data for a three-dimensional line
ax.plot3D(arm_x,arm_y,arm_z, 'red')