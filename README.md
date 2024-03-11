# LeArm Controller and Kinematics Library

## Controller
This controller uses the `hidapi` and hex commands to control LeArm. The `servoMove` command updates the position of servos it has moved. Using a file strcuture of:  

script.py  
Controller  
Kinematics  

The controller can be used with:
```python

from Controller.LeArm import LeArm, LSC_Series_Servo

arm = LeArm(debug=True)

arm.servo1.position = 1500
arm.servo2.position = 1500
arm.servo3.position = 1500
arm.servo4.position = 1500
arm.servo5.position = 1500
arm.servo6.position = 1500

servos = [arm.servo6, arm.servo5, arm.servo4, arm.servo3, arm.servo2]
arm.servoMove(servos, [90, 90, 90, 90, 90], time=2000)
```

## Kinematics
The system uses an orgin placed at the hole in the black base plate as the world frame. The world frame, local frames of all links, and servo/motor orientations, all strictly follow the right hand rule and use units of cm.  
![1567159000471111](https://github.com/thomazach/LeArm/assets/86134403/6fcc7400-7670-4a97-9dac-a12c4bc43a1d)

The starting position of the arm is assumed to be straight upwards. With the same file structure as the controller, various kinematic tools can be accessed.

```python
from Kinematics.kinematics import *
from Kinematics.matrix import matrix

linkVectors, jointAxis, jointAngles = buildArm()  # Arm geometry from arm_attributes.yaml which can be changed
                                                  # to support non-stock mechanical add ons to the arm
newPosAngles = goToPos(linkVectors, jointAxis, jointAngles, matrix([[x],[y],[z]]) # Inverse kinematic call, where x, y, and z is the desired end position in cm
newServoAngles = convertModelAnglesToServoAngles(newPosAngles) # Returns servo angles in degrees

# Get the current servo angles into angles that work with the inverse kinematics:
modelAngles = convertServoAnglesToModelAngles([arm.servo6.giveDegrees(),
                                               arm.servo5.giveDegrees(),
                                               arm.servo4.giveDegrees(),
                                               arm.servo3.giveDegrees(),
                                               arm.servo2.giveDegrees(),
                                               arm.servo1.giveDegrees()]

newPosAngles = goToPos(linkVectors, jointAxis, modelAngles, matrix([[x],[y],[z]]))


```
The `convertModelAnglesToServoAngles` and the `convertServoAnglesToModelAngles` can be adjusted in the event that a servo motor was installed backwards. The source code will need to be edited. Alpha 1 and alpha 2 in this figure 
represent the angles used by the kinematic model.  
![HattonOptimalGaitsFig2Screenshot](https://github.com/thomazach/LeArm/assets/86134403/28f1d22d-b65d-4c75-af4d-3d5d7e9b66e6)  
When coding a custom conversion between this model's angles and servo angles, the conversions in each function should be inverse operations of eachother, and the actual conversion does not scale angles (1 degree in the model's angle is 1 degree in the servo's angles). 
If you're arm has different geometry, (the length of the links has been changed or the orientation of the servos relative to their respective links has changed), then arm_attributes.yaml will need to be adjusted. The `jointAngles` list contains angles in radians,
where the first entry coresponds to the stationary servo. The `linkVectors` list contains 3 column vectors of x, y, and z coordinates in cm, which correspond to the size of the link in its local frame. The first entry in `linkVectors` is an offset
from the origin and represents increasing the distance between servo6 and servo5. It is set to 8.5cm, to lower the coordinate frame to the black plate of LeArm. The local frame orientations of the links follow the right hand rule, with the x-axis extending radially outwards
from the servo's rotating element, and the z-axis extending normal to the face of the rotating servo element. The `jointAxis` list represents the orientation of the motor relative to the previous link, for the first link, its orientation is relative to the world frame. The
orientation of a servo is its axis of rotation (x, y, or z) relative to the previous servo.
