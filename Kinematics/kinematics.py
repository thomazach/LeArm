
import os
import math
import yaml
import random

import numpy as np
np.set_printoptions(precision=2, suppress=True)

import numdifftools as nd
#import matplotlib.pyplot as plt

from matrix import matrix
from mpl_toolkits.mplot3d import Axes3D

import time

def buildArm():

    oldWorkingDir = os.getcwd()
    os.chdir(os.path.dirname(__file__)) # Set this folder to working directory for cross-platform naming conventions (unix and windows)

    with open("arm_attributes.yaml", "r") as f:
        armData = yaml.safe_load(f)
    
    jointAxis = armData['jointAxis']
    jointAngles = armData['jointAngles']

    linkVectors = []
    for link in armData['linkVectors']:
        linkVectors += [matrix(link)]

    os.chdir(oldWorkingDir) # Restore previous working directory, in case this is being imported from another parent folder

    return linkVectors, jointAxis, jointAngles

def rotationSet(jointAxis, jointAngles):
        '''Converts a set of jointAxis and jointAngles into a list of rotation matrices.'''

        # Assign the apropriate rotation matrix to each axis based off its global frame rotation
        R_set = []
        for idx, jointAxis in enumerate(jointAxis):

            cosAlpha = math.cos(jointAngles[idx])
            sinAlpha = math.sin(jointAngles[idx])

            match jointAxis:

                case 'x':
                    R_set.append(matrix([
                           [1, 0, 0],
                           [0, cosAlpha, -sinAlpha],
                           [0, sinAlpha, cosAlpha]
                           ]))
                case 'y':
                    R_set.append(matrix([
                            [cosAlpha, 0, sinAlpha],
                            [0, 1, 0],
                            [-sinAlpha, 0, cosAlpha]
                        ]))
                case 'z':
                    R_set.append(matrix([
                            [cosAlpha, -sinAlpha, 0],
                            [sinAlpha, cosAlpha, 0],
                            [0, 0, 1]
                        ]))
                
                case _:
                    raise ValueError(f"{jointAxis} is not a known joint axis. Use 'x', 'y', or 'z'.")
        
        return R_set

def rotationSetCumulativeProduct(rotationSet):
        '''Input: rotationSet - list of matrix objects representing link orientations relative to the previous link
           Output: List of rotation matrix objects representing link orientations relative to the global frame

           Notes:
           Since each link is connected, the left multiplication of each link's rotation matrices (link_1 * link_2 * link_3 ... link_n) would equal
           the rotational orientation of link n relative to the world frame. This function returns a list of rotation matrices relative to the world 
           frame in the order of each link. So entry 0 of the list corresponds to the rotation matrix representing the global orientation of the first link
           and so on.
        '''

        rotationSetCumulative = rotationSet

        for idx, rotationMatrix in enumerate(rotationSet):
            # First rotation matrix is already in global coordinates
            if idx == 0:
                continue

            rotationSetCumulative[idx] = rotationSetCumulative[idx - 1] * rotationMatrix

        return rotationSetCumulative

def rotateVectors(vectors, rotationMatrices):
        '''Rotate n vectors by n rotation Matrices. Both the vectors and rotation matrices are matrix objects.'''

        # for vector, rotationMatrix, idx in zip(rotatedVectors, rotationMatrices, range(len(vectors))):
        #     vector = rotationMatrix * vectors[idx]

        rotatedVectors = []
        for i, vector in enumerate(vectors):
            rotatedVectors.append(rotationMatrices[i] * vector)
        
        return rotatedVectors

def vectorSetCumulativeSum(vectors):
        '''Input: vectors - List of matrix objects
        Output: vectorsCumulativeSum - List of matrix objects
        
        Function: Takes a set of vectors: [v1, v2, v3, ... vN] and returns a set of vectors: [v1, v1 + v2, v1 + v2 + v3...] 
        '''

        vectorsCumulativeSum = [vectors[0]]

        for idx, vector in enumerate(vectors):
            # Skip the first entry
            if idx == 0:
                continue

            vectorsCumulativeSum += [vectorsCumulativeSum[idx - 1] + vector]

        return vectorsCumulativeSum

def getPosition(linkVectors, jointAxis, jointAngles, graph=False, useRigidBodies=False):
        '''Returns the end point of each link in global coordinates. The end of the first link is the start of the second link.
        The "First link" begins at the connection between the servo6 and the lower rotating plate, this is also the origin of the
        coordinate system.

        Optional Inputs: (WIP)
        graph: bool, default False
            If true it will graph the arms position in 3D using matplot lib
            
        useRigidBodies: bool, default False
            Calculates position as a collection of rigid bodies for each link, and will graph them if graph is True
        '''

        RSet= rotationSet(jointAxis, jointAngles) # Create rotation geometry relative from base link
        rotationSetCumulative = rotationSetCumulativeProduct(RSet) # Create rotation geometry relative to global frame
        vectorsGlobalRotation = rotateVectors(linkVectors, rotationSetCumulative) # Rotate individual vectors to the global frame
        vectorsGlobalFrame = vectorSetCumulativeSum(vectorsGlobalRotation) # Add vectors together so that they each end at the end of their respective link

        return vectorsGlobalFrame

def moveToPosition(linkVectors, jointAxis, jointAngles, vector, distanceFromPos=0.2, linkNumber=-1, jointLimits=[(-90, 90), (-180, 0), (-90, 90), (-90, 90), (-90, 90)]):
    '''vector: Matrix cartesian vector in world frame [[x], [y], [z]]
    distanceFromPos: Acceptable amount of displacement from requested position 

        Produces a set of joint angles that place the end effector at the tip of vector.
        If no possible configuration exists to produce the end effector position, a warning will
        be printed and the funcition will return the current position of the arm.
    '''

    if np.linalg.norm(vector.mat) > 44.8:
        raise ValueError(f"The requested position vector {vector} is outside the reach of the arm.")

    def endPointDifferenceMap(jointAngles):

        for i, jointLimit in enumerate(jointLimits):
             
            if min(jointLimit) > math.degrees(jointAngles[i]):
                jointAngles[i] = math.radians(min(jointLimit))
            elif max(jointLimit) < math.degrees(jointAngles[i]):
                jointAngles[i] = math.radians(max(jointLimit))
                  
        
        endEffectorPos = getPosition(linkVectors, jointAxis, jointAngles)[linkNumber]
        difference = endEffectorPos - vector
        absDifference = np.linalg.norm(difference.mat)

        return absDifference
    
    endPointDifferenceMapGradient = nd.Gradient(endPointDifferenceMap)

    zeroVector = endPointDifferenceMap(jointAngles)

    while zeroVector > distanceFromPos:
        gradient = endPointDifferenceMapGradient(jointAngles)
        negativeGradient = gradient * -1

        tempJointAngles = []
        for jointAngle, gradientParameter in zip(jointAngles, negativeGradient):
            tempJointAngles += [jointAngle + gradientParameter * (0.005 + random.uniform(0, 0.0001))]
        jointAngles = tempJointAngles

        zeroVector = endPointDifferenceMap(jointAngles)
        #print(zeroVector)

    return jointAngles

def convertModelAnglesToServoAngles(modelJointAngles):
    '''MODEL JOINT ANGLES IN RADIANS'''

    servoJointAngles = [None] * len(modelJointAngles)
     
    for i, modelJointAngle in enumerate(modelJointAngles):
        
          
        match i:
            case 0 | 2 | 3 | 4:
                servoJointAngles[i] = math.degrees(modelJointAngle) + 90
            case 1:
                servoJointAngles[i] = math.degrees(modelJointAngle) * -1

    return servoJointAngles

def convertServoAnglesToModelAngles(servoJointAngles):
    '''SERVO ANGLE IN DEGREES'''
     
    modelAngles = [None] * len(servoJointAngles)

    for i, servoAngle in enumerate(servoJointAngles):
         
        match i:
            case 0 | 2 | 3 | 4:
                modelAngles[i] = math.radians(servoAngle - 90)
            case 1:
                modelAngles[i] = math.radians(servoAngle * -1)
    
    return modelAngles

def motionPlan(linkVectors, jointAxis, jointAngles, endPos):
    print(jointAngles)
    startingPos = getPosition(linkVectors, jointAxis, jointAngles)[-1]
    print(startingPos)

    x = np.linspace(startingPos.mat[0][0], endPos.mat[0][0], 10)
    y = np.linspace(startingPos.mat[1][0], endPos.mat[1][0], 10)
    z = np.linspace(startingPos.mat[2][0], endPos.mat[2][0], 10)

    #print(x)
    #print(y)
    #print(z)

    motionPlanAngles = []
    for i, xPos in enumerate(x):
        jointAngles = moveToPosition(linkVectors, jointAxis, jointAngles, matrix([[xPos], [y[i]], [z[i]]]))
        print(f"Planned motion step {i} of 9")
        motionPlanAngles.append(convertModelAnglesToServoAngles(jointAngles))
    
    return motionPlanAngles
        


          
     
# linkVectors, jointAxis, jointAngles = buildArm()
# print("Joint angles in the start position: ", jointAngles)


# startingEndPoints = getPosition(linkVectors, jointAxis, jointAngles)
# startingPos = startingEndPoints[-1]
# print("End effector at the initial joint angles: ", startingPos)
# print()
# desiredEndPos = matrix([[11.5], [-12], [15.5]])
# print("Requested final position of end effector: ", desiredEndPos)
# print("Difference of magnitude between requested position and simulated position:")

# #time.sleep(2)

# newJointAngles = moveToPosition(linkVectors, jointAxis, jointAngles, desiredEndPos, distanceFromPos = 0.1)

# print("Joint angles (model) resulting from manifold gradient descent: ", newJointAngles)
# print("Joint angles (servo degrees): ", convertModelAnglesToServoAngles(newJointAngles))

# newLinkEnds =  getPosition(linkVectors, jointAxis, newJointAngles)
# newEndPoint = newLinkEnds[-1]
# print("New joint angles result in a final end effector position of: \n", newEndPoint)
# print()

# error = newEndPoint - desiredEndPos
# print("Error in cm: ", error)

# # Create graph
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# startVector =[[0], [0], [0]]
# for endVector in newLinkEnds:
#     x = np.linspace(startVector[0][0], endVector.mat[0][0])
#     y = np.linspace(startVector[1][0], endVector.mat[1][0])
#     z = np.linspace(startVector[2][0], endVector.mat[2][0])


#     startVector = endVector.mat
#     ax.plot(x, y, z)

# # Stackoverflow copy pasta for axis equal
# x_limits = ax.get_xlim3d()
# y_limits = ax.get_ylim3d()
# z_limits = ax.get_zlim3d()

# x_range = abs(x_limits[1] - x_limits[0])
# x_middle = np.mean(x_limits)
# y_range = abs(y_limits[1] - y_limits[0])
# y_middle = np.mean(y_limits)
# z_range = abs(z_limits[1] - z_limits[0])
# z_middle = np.mean(z_limits)

# # The plot bounding box is a sphere in the sense of the infinity
# # norm, hence I call half the max range the plot radius.
# plot_radius = 0.5*max([x_range, y_range, z_range])

# ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
# ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
# ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# plt.show()


# # From servo-motor reported degrees:
#     # For servos 6, 4, & 3: joint angle = servo-motor reported degrees - 90
#     # For servo 5: joint angle = servo-motor reported degrees * -1