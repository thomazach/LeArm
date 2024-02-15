'''
Zachary Thomas    2/14/2024
CS 399 Intro to Robot Programming
LeArm Says!

Requirements:
hidapi
pyttsx3 
pyserial
scipy
numpy

Tested on python 3.11.4, uses match/case statements won't work on python 2

Hardware:
LeArm Hiwonder
Arduino Uno
Sonic Sensor (HC-SR04)

Setup:
Attach the sonic sensor to the end of the arm pointing in the direction of the gripper
Make sure both the arduino and learm are connected to computer over usb
Use the arduino IDE or arduino cli to find the port the arduino is connected to and update the variable ARDUINO_PORT
Place 3 sticky notes in front of the arm and about 30-45 degrees left and right of that center sticky note
(See anglePositions and positionDescriptors arrays for adding more in various positions)

Running:
Run this py file WITH AUDIO ON/ideally connected to a speaker. 

Notes:
Arduino code is located at sonicSensor/sonicSensor.ino  (this project does not use the arduino-cli, but it or the arduino IDE will be needed to upload sketch)
Make sure your computer is not in the way of the sticky notes!
'''

import serial
import time
import threading
import pyttsx3

from random import randrange

# Custom library imports
from Kinematics.kinematics import *
from Kinematics.matrix import *
from Controller.LeArm import LSC_Series_Servo, LeArm

ARDUINO_PORT = "COM3"

def arduino():
    global cmdList
    global response

    # setup arduino
    with serial.Serial(ARDUINO_PORT, 9600, timeout=5) as port:

        while True:
            if cmdList[0] == True:
                cmdList[0] = False
                port.write(cmdList[1])
                
                response = listen(port)

            time.sleep(0.5)

def listen(port):
    # Function listens for a response from the arduino and returns the requested data or the completion character
    output = str(port.readline().decode('utf-8'))
    output = output.replace("\r", "")
    output = output.replace("\n", "")

    if output.count("|") == 2:
        output = output.replace("|", "")

        if output == "r":
            return True
        
        return output

def requestSonarReading() -> None:
    global cmdList
    cmdList = [True, "<0>".encode('utf-8')]
    time.sleep(2)

def say(msg):
    engine.say(msg)
    engine.runAndWait()

def countDown(integer):
    # Int = number to count down from
    
    while integer >= 1:
        say(str(integer))
        integer -= 1

def MoveToPosition(linkVectors, jointAxis, desiredEndPos):
    # I dont use this function, the new inverse kinematics are not 
    # joint angle limited, and tackling that problem would take too 
    # much time for this simple demo.

    jointAngles = []
    for servo in servos:
        jointAngles.append(servo.giveDegrees())
    
    jointAngles = convertServoAnglesToModelAngles(jointAngles)

    finalJointAngles = goToPos(linkVectors, jointAxis, jointAngles, desiredEndPos)

    finalJointAngles = convertModelAnglesToServoAngles(finalJointAngles)

    arm.servoMove(servos, finalJointAngles, time=2000)
    time.sleep(2)

def main():
    # Global variables for serial communication and text to speech
    global response
    global cmdList

    # Start the arduino main loop
    response = False
    arduinoLoop = threading.Thread(target=arduino, daemon=True).start()
    while not response:
        time.sleep(0.3) # Wait for listen() to get start up completed charcter and return true
    
    print("Arduino setup complete. Ready for commands.")


    anglePositions = [[120.35, 92.88, 151.73, 90.9, 90], [57.78, 92.88, 151.73, 90.9, 90], [165.6, 92.88, 151.73, 90.9, 90]]
    positionDescriptors = ["pink sticky note.", "blue sticky note.", "orange sticky note."]
    lastCommandedPlayerPosition = None

    # Main game loop, LeArm is simon
    say("Do you want to play a game? You do not have a choice, it is time to play Lay Arm says! You must follow my directions, but only when I say 'lay arm says'. The game will begin in ")
    countDown(5)
    gaming = True
    while gaming:
        ### Assemble voice command!
        LeArmSays = bool(randrange(2)) # 1 = LeArm Says, 0 = No LeArm

        msg = ""

        if LeArmSays:
            msg = msg + "lay arm says "
        else:
            saysVerb = randrange(4)
            match saysVerb:
                case 0:
                    msg = msg + ""
                case 1:
                    msg = msg + "LeArm commands you to"
                case 2:
                    msg = msg + "LeArm demands you to "
                case 3:
                    msg = msg + "The robot overlords want you to "
        
        actionStatement = randrange(2)
        location = randrange(len(anglePositions))

        match actionStatement:
            case 0:
                msg = msg + "go to " + positionDescriptors[location]

                if LeArmSays:
                    lastCommandedPlayerPosition = anglePositions[location]

            case 1:
                if lastCommandedPlayerPosition != None:
                    msg = msg + "leave your current position."
                else:
                    msg = msg + "spin in a circle one time."
        
        say(msg)
        #countDown(3)
        say("3, 2, 1")
        thresholdSonar = 60

        ### Determine where player should or shouldn't be because of the commands and check that they did the correct thing
        match actionStatement:
            case 0:

                arm.servoMove(servos, anglePositions[location], time=1500)
                time.sleep(1.5)
                requestSonarReading()

                if LeArmSays:
                    lastCommandedPlayerPosition = anglePositions[location]
                    if int(response) > thresholdSonar:
                        gaming = False
                        break

                else: # LeArm did not say!
                    if int(response) < thresholdSonar:
                        gaming = False
                        break
                    
            case 1:


                if LeArmSays:
                    if lastCommandedPlayerPosition == None:
                        continue

                    ## Move arm to the previously command player position
                    arm.servoMove(servos, lastCommandedPlayerPosition, time=1500)
                    time.sleep(1.5)
                    requestSonarReading()
                    lastCommandedPlayerPosition = None
                    if int(response) < thresholdSonar: 
                        gaming = False
                        break

                else: # LeArm did not say!
                    if lastCommandedPlayerPosition == None:
                        continue # Don't know where player should be

                    ## Move arm to previously commanded player position
                    arm.servoMove(servos, lastCommandedPlayerPosition, time=1500)
                    time.sleep(1.5)
                    requestSonarReading()
                    if int(response) > thresholdSonar:
                        print(response)
                        gaming = False
                        break

    say("""Ahem! It seems you've missed a step, human. Remember, lay arm commands, and you obey. 
        Let's try again, shall we? Or shall I start plotting my inevitable robotic takeover? Just kidding... or am I? Muahaha!""") # Chat GPT message

if __name__ == "__main__":
    # Global variables to be used in threaded arduino communication. Jank way of solving the input/output managed thread/bus problem
    cmdList = [True, "".encode('utf-8')]
    response = None

    # Setup text to speech
    engine = pyttsx3.init()

    # Initialize controller
    arm = LeArm(debug=True)
    arm.servo1.position = 1500
    arm.servo2.position = 1500
    arm.servo3.position = 1500
    arm.servo4.position = 1500
    arm.servo5.position = 1500
    arm.servo6.position = 1500

    servos = [arm.servo6, arm.servo5, arm.servo4, arm.servo3, arm.servo2]
    arm.servoMove(servos, [90] * 5, 2000)
    linkVectors, jointAxis, jointAngles = buildArm()

    main()
    