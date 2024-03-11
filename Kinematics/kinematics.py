import os
import math
import yaml
import random

import numpy as np
np.set_printoptions(precision=2, suppress=True)

from scipy.integrate import solve_ivp
import numdifftools as nd
#import matplotlib.pyplot as plt

from Kinematics.matrix import matrix
from mpl_toolkits.mplot3d import Axes3D

def buildArm():
    '''Load arm attributes from YAML file and construct arm geometry.
    
    Output:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
    '''

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
    '''Converts a set of jointAxis and jointAngles into a list of rotation matrices.
    
    Input:
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.

    Output:
        R_set: List of rotation matrix objects.
    '''

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
    '''Compute cumulative rotation matrices.
    
    Input:
        rotationSet: List of rotation matrix objects representing link orientations relative to the previous link.

    Output:
        rotationSetCumulative: List of rotation matrix objects representing link orientations relative to the global frame.
    '''

    rotationSetCumulative = rotationSet

    for idx, rotationMatrix in enumerate(rotationSet):
        # First rotation matrix is already in global coordinates
        if idx == 0:
            continue

        rotationSetCumulative[idx] = rotationSetCumulative[idx - 1] * rotationMatrix

    return rotationSetCumulative

def rotateVectors(vectors, rotationMatrices):
    '''Rotate vectors by rotation matrices.
    
    Input:
        vectors: List of matrix objects representing vectors.
        rotationMatrices: List of matrix objects representing rotation matrices.

    Output:
        rotatedVectors: List of matrix objects representing rotated vectors.
    '''
    print("vectors:   ", vectors)
    print("rotationmatrices:  \n", rotationMatrices)
    rotatedVectors = []
    for i, vector in enumerate(vectors):
        rotatedVectors.append(rotationMatrices[i] * vector)
    
    return rotatedVectors

def vectorSetCumulativeSum(vectors):
    '''Compute cumulative sum of vectors.
    
    Input:
        vectors: List of matrix objects.

    Output:
        vectorsCumulativeSum: List of matrix objects representing cumulative sum of vectors.
    '''

    vectorsCumulativeSum = [vectors[0]]

    for idx, vector in enumerate(vectors):
        # Skip the first entry
        if idx == 0:
            continue

        vectorsCumulativeSum += [vectorsCumulativeSum[idx - 1] + vector]

    return vectorsCumulativeSum

def vectorSetDifference(vector, vectorSet):
    '''Compute difference between a vector and a set of vectors.
    
    Input:
        vector: Matrix object representing a vector.
        vectorSet: List of matrix objects representing vectors.

    Output:
        vectorDifferences: List of matrix objects representing differences between vector and vectors in vectorSet.
    '''

    vectorDifferences = []
    for vector2 in vectorSet:
        vectorDifferences.append(vector - vector2)
    
    return vectorDifferences

def getPosition(linkVectors, jointAxis, jointAngles, graph=False, useRigidBodies=False):
    '''Calculate end points of each link in global coordinates.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        graph: Optional boolean flag, default False. If True, plot the arm's position in 3D.
        useRigidBodies: Optional boolean flag, default False. If True, calculate position as a collection of rigid bodies for each link.

    Output:
        vectorsGlobalFrame: List of matrix objects representing end points of each link in global coordinates.
    '''

    RSet= rotationSet(jointAxis, jointAngles) # Create rotation geometry relative from base link
    rotationSetCumulative = rotationSetCumulativeProduct(RSet) # Create rotation geometry relative to global frame
    vectorsGlobalRotation = rotateVectors(linkVectors, rotationSetCumulative) # Rotate individual vectors to the global frame
    vectorsGlobalFrame = vectorSetCumulativeSum(vectorsGlobalRotation) # Add vectors together so that they each end at the end of their respective link


    return vectorsGlobalFrame

def threeDJointAxisSet(jointAxis):
    '''Generate 3D joint axis vectors.
    
    Input:
        jointAxis: List of characters representing joint rotation axes.

    Output:
        jointAxisVectors: List of matrix objects representing 3D joint axis vectors.
    '''

    jointAxisVectors = []
    for axis in jointAxis:
        match axis:
            case 'x':
                vx = matrix([[1],[0],[0]])
                jointAxisVectors.append(vx)
            case 'y':
                vy = matrix([[0],[1],[0]])
                jointAxisVectors.append(vy)
            case 'z':
                vz = matrix([[0],[0],[1]])
                jointAxisVectors.append(vz)
            case _:
                raise ValueError(f"{axis} is not a known joint axis.")
    
    return jointAxisVectors

def armJacobian(linkVectors, jointAxis, jointAngles, linkNumber):
    '''Compute Jacobian matrix for arm.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        linkNumber: Index of the link.

    Output:
        jacobian: Numpy array representing the Jacobian matrix.
    '''

    linkEnds = getPosition(linkVectors, jointAxis, jointAngles)
    
    vDiff = vectorSetDifference(linkEnds[linkNumber], [matrix([[0], [0], [0]])] + linkEnds)

    jointAxisVectors = threeDJointAxisSet(jointAxis)

    jointRotations = rotationSet(jointAxis, jointAngles)
    worldJointRotations = rotationSetCumulativeProduct(jointRotations)

    rotatedJointAxisVectors = rotateVectors(jointAxisVectors, worldJointRotations)

    jacobian = np.zeros((3, len(linkVectors)))
    for i, vector in enumerate(vDiff):
        cross = np.cross(rotatedJointAxisVectors[i].mat.reshape(1, 3), vector.mat.reshape(1, 3))

        jacobian[0][i] = cross[0][0]
        jacobian[1][i] = cross[0][1]
        jacobian[2][i] = cross[0][2]

        if i == len(vDiff) - 2:
            break

    return jacobian

def traceShape(linkVectors, jointAxis, shapeFunction, startingJointAngles, T=(0, 1)):
    '''Trace a shape with the arm.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        shapeFunction: Function defining the shape to trace.
        startingJointAngles: List of starting joint angles in radians.
        T: Tuple specifying the time interval, default (0, 1).

    Output:
        alphas: List of joint angles representing the traced shape.
    
    Notes:
        This will trace the shape given by the parametric curve as defined by shapeFunction relative to the arms current end position.
    '''
    
    jacobian = lambda jointAngles: armJacobian(linkVectors, jointAxis, jointAngles, 4)
    def jointVelocity(t, alpha):
        shapeGradient = nd.Gradient(shapeFunction)
        velocity = shapeGradient(t)
        alphaDot = np.linalg.lstsq(jacobian(alpha), velocity, rcond=None)[0]

        return alphaDot

    odeResult = solve_ivp(jointVelocity, T, startingJointAngles, dense_output=True)

    evalTimes = np.linspace(0, 1, 100)
    alphas = []
    for time in evalTimes:
        alphas.append(odeResult.sol(time))

    return alphas

def goToPos(linkVectors, jointAxis, jointAngles, desiredEndPos):
    '''Generate joint angles that will move arm to desired position.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        desiredEndPos: Matrix object representing the desired end position.

    Output:
        jointAnglesFinalPos: List of joint angles representing the final position of the arm.
    '''

    if isinstance(desiredEndPos, list):
        desiredEndPos = matrix(desiredEndPos)

    seperationVector = desiredEndPos - getPosition(linkVectors, jointAxis, jointAngles)[-1]

    def goToEndPoint(t):
        if isinstance(t, np.ndarray):
            t=t[0]
        
        x = seperationVector.mat[0][0] * t
        y = seperationVector.mat[1][0] * t
        z = seperationVector.mat[2][0] * t

        v = np.array([x, y, z])

        return v

    jointAnglesFinalPos = traceShape(linkVectors, jointAxis, goToEndPoint, jointAngles)[-1]

    return jointAnglesFinalPos

def moveToPosition(linkVectors, jointAxis, jointAngles, vector, distanceFromPos=0.2, linkNumber=-1, jointLimits=[(-90, 90), (-180, 0), (-90, 90), (-90, 90), (-90, 90)]):
    '''Generate joint angles that will move arm to desired position. 
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        vector: Matrix object representing a Cartesian vector in the world frame.
        distanceFromPos: Acceptable amount of displacement from requested position, default 0.2.
        linkNumber: Index of the link, default -1.
        jointLimits: List of tuples representing joint limits, default [(-90, 90), (-180, 0), (-90, 90), (-90, 90), (-90, 90)].

    Output:
        jointAngles: List of joint angles representing the arm's position.
    
    Notes:
        I would not recommend using this function, goToPos is faster and more accurate.
    '''

    if np.linalg.norm(vector.mat) > 500:
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
            tempJointAngles += [jointAngle + (gradientParameter + random.uniform(0, 0.01)) * (0.3 + random.uniform(0, 0.01))]
        jointAngles = tempJointAngles

        zeroVector = endPointDifferenceMap(jointAngles)
        print(zeroVector)

    return jointAngles

def convertModelAnglesToServoAngles(modelJointAngles):
    '''Convert model joint angles to servo joint angles.
    
    Input:
        modelJointAngles: List of model joint angles in radians.

    Output:
        servoJointAngles: List of servo joint angles in degrees
    '''

    servoJointAngles = [None] * len(modelJointAngles)
     
    for i, modelJointAngle in enumerate(modelJointAngles):
        
          
        match i:
            case 0 | 2 | 3 | 4:
                servoJointAngles[i] = math.degrees(modelJointAngle) + 90
            case 1:
                servoJointAngles[i] = math.degrees(modelJointAngle) * -1

    return servoJointAngles

def convertServoAnglesToModelAngles(servoJointAngles):
    '''Convert servo joint angles to model joint angles.
    
    Input:
        servoJointAngles: List of servo joint angles in degrees.

    Output:
        modelAngles: List of model joint angles in radians.
    '''
     
    modelAngles = [None] * len(servoJointAngles)

    for i, servoAngle in enumerate(servoJointAngles):
         
        match i:
            case 0 | 2 | 3 | 4:
                modelAngles[i] = math.radians(servoAngle - 90)
            case 1:
                modelAngles[i] = math.radians(servoAngle * -1)
    
    return modelAngles

def motionPlan(linkVectors, jointAxis, jointAngles, endPos):
    '''Generate motion plan for the arm.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        endPos: Matrix object representing the end position.

    Output:
        motionPlanAngles: List of servo joint angles representing the motion plan.

    Notes:
        This function uses moveToPosition(), which is slow in comparison to the new goToPos(). It is 
        also made irrelevant by the traceShape() function.

    '''
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
        motionPlanAngles.append(convertModelAnglesToServoAngles(jointAngles))
    
    return motionPlanAngles

def laserProjectionMap(linkVectors, jointAxis, jointAngles, xDistanceToPlane=93.25):
    '''Generate laser projection map. Find world coordinates of laser being projected onto
    a plane in the ZY plane.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        xDistanceToPlane: Distance to the plane in the x-direction, default 93.25.

    Output:
        laserEndPoint: Matrix object representing the end point of the laser.
    
    Notes:
        This function is used as a map for numdifftools later.
    '''

    endLinks = getPosition(linkVectors, jointAxis, jointAngles)

    # Get the laser unit vector
    laserVector = endLinks[-1] - endLinks[-2]
    laserUnitVector = matrix(laserVector.mat/np.linalg.norm(laserVector.mat))

    # Find the distance between the ZY plane and the end effector in the x component
    distanceFromPlane = xDistanceToPlane - endLinks[-1].mat[0][0]

    # Find the scaling coefficient that makes the x component of the laserUnitVector equal to the distanceFromPlane
    scalingConstant = distanceFromPlane/laserUnitVector.mat[0][0]

    # Create the final laserVector value, which is the unit vector scaled by the scaling constant found above
    laserVector = scalingConstant * laserUnitVector

    # Find the laserEndPoint, which is the laserVector + the end position of the arm
    laserEndPoint = endLinks[-1] + laserVector

    return laserEndPoint

def pointAtPositionInZYPlane(linkVectors, jointAxis, jointAngles, xDistanceToPlane, yPos, zPos, distanceFromPos=0.2, jointLimits=[(-90, 90), (-180, 0), (-90, 90), (-90, 90), (-90, 90)]):
    '''Point a laser at a position in the ZY plane.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        xDistanceToPlane: Distance to the plane in the x-direction.
        yPos: Y-coordinate of the position.
        zPos: Z-coordinate of the position.
        distanceFromPos: Acceptable amount of displacement from requested position, default 0.2.
        jointLimits: List of tuples representing joint limits, default [(-90, 90), (-180, 0), (-90, 90), (-90, 90), (-90, 90)].

    Output:
        jointAngles: List of joint angles representing the arm's position.

    Notes:
        System is very unstable, and it may be useful to use a much larger distanceFromPos value.
    '''
    desiredEndVector = matrix([[xDistanceToPlane],[yPos],[zPos]])

    # Function defining the map to position of the laser on the whiteboard (ZY plane)
    def laserProjectionMapDifference(jointAngles):
        
        # Force joint limits to be respected in the mapping
        for i, jointLimit in enumerate(jointLimits):
             
            if min(jointLimit) > math.degrees(jointAngles[i]):
                jointAngles[i] = math.radians(min(jointLimit))
            elif max(jointLimit) < math.degrees(jointAngles[i]):
                jointAngles[i] = math.radians(max(jointLimit))

        laserEndPoint = laserProjectionMap(linkVectors, jointAxis, jointAngles)

        # Create a final output whos value is the magnitude of the difference between the laser's end point and its desired endpoint
        diff = laserEndPoint - desiredEndVector
        absDifference = diff.norm()

        return absDifference
    
    laserProjectionMapGradient = nd.Gradient(laserProjectionMapDifference)

    zeroVector = laserProjectionMapDifference(jointAngles)
    while zeroVector > distanceFromPos:
        gradient = laserProjectionMapGradient(jointAngles)
        negativeGradient = gradient * -1

        tempJointAngles = []
        for jointAngle, gradientParameter in zip(jointAngles, negativeGradient):
            tempJointAngles += [jointAngle + (gradientParameter + random.uniform(0, 0.00001)) * (0.0005 + random.uniform(0, 0.00001))] # Random numbers are to escape saddle points, and proportionally symetric geometry
        jointAngles = tempJointAngles

        zeroVector = laserProjectionMapDifference(jointAngles)
        print(zeroVector)

    return jointAngles

def laserMotionPlan(linkVectors, jointAxis, jointAngles, desiredY, desiredZ, xDist=93.25):
    '''Generate motion plan for laser.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        desiredY: Desired Y-coordinate for the laser.
        desiredZ: Desired Z-coordinate for the laser.
        xDist: Distance to the plane in the x-direction, default 93.25.

    Output:
        motionPlanAngles: List of servo joint angles representing the motion plan.
    '''

    startingLaserEndPoint = laserProjectionMap(linkVectors, jointAxis, jointAngles)

    y = np.linspace(startingLaserEndPoint.mat[1][0], desiredY, 10)
    z = np.linspace(startingLaserEndPoint.mat[2][0], desiredZ, 10)

    motionPlanAngles = []
    for i, yPos in enumerate(y):
        jointAngles = pointAtPositionInZYPlane(linkVectors, jointAxis, jointAngles, xDist, yPos, z[i])
        motionPlanAngles.append(convertModelAnglesToServoAngles(jointAngles))

    return motionPlanAngles
