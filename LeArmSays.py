import serial
import time
import threading
import pyttsx3

from random import randrange

# Custom library imports
from Kinematics.kinematics import *
from Kinematics.matrix import *
from Controller.LeArm import LSC_Series_Servo, LeArm

def arduino():
    global cmdList
    global response

    # setup arduino
    with serial.Serial("COM4", 9600, timeout=5) as port:

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


    cartesianPositions = [0, 1, 2, 3]
    positionDescriptors = ["red sticky note.", "pink sticky note.", "blue sticky note.", "green sticky note."]
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
        location = randrange(len(cartesianPositions) - 1)

        match actionStatement:
            case 0:
                msg = msg + "go to " + positionDescriptors[location]

                if LeArmSays:
                    lastCommandedPlayerPosition = cartesianPositions[location]

            case 1:
                if lastCommandedPlayerPosition != None:
                    msg = msg + "leave your current position."
                else:
                    msg = msg + "spin in a circle one time."
        
        say(msg)
        countDown(3)
        thresholdSonar = 60

        ### Determine where player should or shouldn't be because of the commands and check that they did the correct thing
        match actionStatement:
            case 0:

                MoveToPosition(linkVectors, jointAxis, cartesianPositions[location])
                requestSonarReading()

                if LeArmSays:
                    if int(response) > thresholdSonar:
                        gaming = False
                        break

                else: # LeArm did not say!
                    if int(response) < thresholdSonar:
                        gaming = False
                        break
                    
            case 1:


                if LeArmSays:
                    ## Move arm to the previously command player position
                    MoveToPosition(linkVectors, jointAxis, lastCommandedPlayerPosition)
                    requestSonarReading()
                    lastCommandedPlayerPosition = None
                    if int(response) < thresholdSonar: 
                        gaming = False
                        break

                else: # LeArm did not say!
                    if lastCommandedPlayerPosition == None:
                        continue # Don't know where player should be

                    ## Move arm to previously commanded player position
                    MoveToPosition(linkVectors, jointAxis, lastCommandedPlayerPosition)
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