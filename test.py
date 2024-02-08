import time

from Controller.LeArm import LeArm, LSC_Series_Servo
from Kinematics.kinematics import *

from Vision import learmVisionV2 as pp 
from Vision import pathfind

arm = LeArm(debug=True)

print("Bat voltage: ", arm.getBatteryVoltage())

arm.servo1.position = 1500
arm.servo2.position = 1500
arm.servo3.position = 1500
arm.servo4.position = 1500
arm.servo5.position = 1500
arm.servo6.position = 1500

servos = [arm.servo6, arm.servo5, arm.servo4, arm.servo3, arm.servo2]
linkVectors, jointAxis, jointAngles = buildArm()

# Reset
arm.servoMove(servos, [90, 90, 90, 90, 90], time=2000)
time.sleep(3)

# Go to maze starting position
arm.servoMove(servos, [107.82, 57.33, 0.0, 96.38999999999999, 90.0], time=1000)
time.sleep(3)

def left(deg=3.6):
    lefterAngle = arm.servo6.giveDegrees() + deg
    arm.servoMove([arm.servo6], angles=[lefterAngle], time=750)

def right(deg=3.6):
    righterAngle = arm.servo6.giveDegrees() - deg
    arm.servoMove([arm.servo6], angles=[righterAngle], time=700)

def up(deg=6):
    upAngle = [arm.servo5.giveDegrees() - deg/2]
    upAngle += [arm.servo3.giveDegrees() + deg/2]
    
    arm.servoMove([arm.servo5, arm.servo3], upAngle, time=750)

def down(deg=6):
    downAngle = [arm.servo5.giveDegrees() + deg/2]
    downAngle += [arm.servo3.giveDegrees() - deg/2]
    arm.servoMove([arm.servo5, arm.servo3], downAngle, time=750)

def getCurrentServoAngles():
    angles = []
    for servo in servos:
        angles.append(servo.giveDegrees())

    return angles
    
def getCurrentLaserPosition(angles):
    angles = convertServoAnglesToModelAngles(angles)
    laserEndPoint = laserProjectionMap(linkVectors, jointAxis, angles)

    return laserEndPoint


On = True
while On:


    path, xy = pathfind.find_maze_solution()
    print(path)
    nextDirection = pathfind.pain_and_anguish(xy, path)

    match nextDirection:
        case 2:
            up()
        case 0:
            left()
        case 1:
            down()
        case 3:
            right()
        case 'info':
            angles = getCurrentServoAngles()
            print("angles= ", angles)
            print(getCurrentLaserPosition(angles))
        case 69430:
            On = False
            

