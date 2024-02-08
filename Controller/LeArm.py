import hid
from time import perf_counter, sleep

class LSC_Series_Servo:
    '''
    Based off of testing on two arms, the servos don't have a way of sending their current position back to the computer.
    This means that PID is impossible, and that there needs to be strict book keeping on where the servo was last told to go.
    The purpose of this object is to handle and store that information. It would be best not to manually change the values of
    object instances and let the LeArm class handle it based off of set position calls.
    '''

    def __init__(self, servoID, limit='default', startingAngle='yaml'):
        '''
        Assign object a servoID matching the arm's naming convention for servos.
        Initialize the position attribute either from the yaml or provide a starting angle in degrees (self.position is in servo units ex. 2000)
        '''
        self.servoID = servoID 
        self.position = None # TODO: Implement startingAngle initialization
        self.movingUntil = -1 # Time in seconds before arm reaches position. Starts at -1 so that it will work the first time, but is changed by self.servoMove()
        self.angularVelocity = None
        self.minimumStep = 10 # Minimum step required for motor to move

        if limit == 'default':
            match self.servoID:
                case 1:
                    self.limit = (1500, 2500)
                case 2 | 3 | 4 | 5 | 6:
                    self.limit = (500, 2500)
                case _:
                    raise ValueError("Servo ID must be an integer between 1 and 6")
        elif isinstance(limit, tuple):
            if len(limit) == 2:
                self.limit = limit
            else:
                raise ValueError("If not 'default' limit must be specified as a 2-element tuple (minLimit, maxLimit)")
        
    def giveDegrees(self):

        # Convert from servo position numbers to degrees
        # Special case, motor is weird with my LeArm
        #if self.servoID == 6:
         #   return 152/(self.limit[1]-self.limit[0]) * self.position - 152/(self.limit[1]-self.limit[0]) * self.limit[0]

        return 180/(self.limit[1]-self.limit[0]) * self.position - 180/(self.limit[1]-self.limit[0]) * self.limit[0]

class LeArm:

    # Command hex codes
    CMD_SERVO_MOVE = 0x03
    CMD_ACTION_GROUP_RUN = 0x06
    CMD_ACTION_GROUP_STOP = 0x07
    CMD_ACTION_SPEED = 0x0B 
    CMD_GET_BATTERY_VOLTAGE = 0x0F # Yes, it really is the same command code as the last one
    CMD_MULT_SERVO_UNLOAD = 0x14
    CMD_MULT_SERVO_POS_READ = 0x15 # This didn't work on the two arms that were tested

    HEADER = 0x55 # Header hex code, controller requires 2x header start bytes to begin reading 

    def __init__(self, debug=False):
        '''
        Currently only tested on windows. On windows the LeArm is a HID device, on linux it will
        be connected to a serial port, and PySerial should be used to control it. Currently I only
        plan on working on windows, so that isn't supported.

        Hardware interfacing based off of this PDF: 
        https://github.com/alexeden/serial-servo/blob/master/docs/LX-16A%20Bus%20Servo/LSC%20Series%20Servo%20Controller%20Communication%20Protocol%20V1.2.pdf

        '''
        self.debug = debug

        self.HIDDevice = hid.device()

        self.servo1 = LSC_Series_Servo(1)
        self.servo2 = LSC_Series_Servo(2)
        self.servo3 = LSC_Series_Servo(3)
        self.servo4 = LSC_Series_Servo(4)
        self.servo5 = LSC_Series_Servo(5)
        self.servo6 = LSC_Series_Servo(6)

        self.HIDDevice.open(0x0483, 0x5750)
        self.HIDDevice.set_nonblocking(1)

        if debug:
            print("Opened HID device of LeArm.")

    def servoMove(self, servos, angles, time=None, velocities=None):
        '''
        Moves LeArm's servos based off of specified inputs. Updates the respective LSC_Series_Servo instances
        with both position, velocity, and termination time. 

        Inputs:
            servos: list of LSC_Series_Servo instances
            angles: list of angles in degrees of each servo
            times: list of times in ms as int
            velocities: list of angular velocities in deg/second
            Note: All lists must be the same length
        '''

        # Convert degree angles into servo values
        servoAngles = []
        for i, servo in enumerate(servos):

            servoAngles += [int(angles[i] * (servo.limit[1]-servo.limit[0])/180 + servo.limit[0])]

        # Check that requested angle is within servo limits
        for servo, angle in zip(servos, servoAngles):

            if angle >= servo.limit[0] and angle <= servo.limit[1]:
                pass
            else:
                raise ValueError(f"The angle for servo {servo.servoID} is {angle} which is outside of this servos limits: {servo.limit}.")
        
        


        # Assemble a hex packet for each time
        numberServos = len(servos)
        timeHexList = self.convertIntToHighLowHex(time)

        params = [numberServos] + timeHexList

        for i, servo in enumerate(servos):
            params += self.generateServoSubParameter(servo.servoID, servoAngles[i])
            

        # Update LSC_Series_Servo instances to have the correct information
        for i, servo in enumerate(servos):
            servo.position = servoAngles[i]
            #servo.angularVelocity = velocities[i]
            servo.movingUntil = int(perf_counter() * 1000) + time
        
        self.send(self.CMD_SERVO_MOVE, params)

    def generateServoSubParameter(self, servoID, angle):
        '''Generate the subparameters of the CMD_SERVO_MOVE hex command. This returns parameters 4, 5, 6 for a given servoID (int) and angle (in servo positional numbers)'''
        angleHexList = self.convertIntToHighLowHex(angle)

        return [servoID] + angleHexList

    def closeHand(self, time=None, velocity=None):
        self.servoMove([self.servo1], [160], time=time, velocities=velocity)
    
    def openHand(self, time=None, velocity=None):
        self.servoMove([self.servo1], [0], time=time, velocities=velocity)

    def servoUnload(self, servos):
        params = []
        for servo in servos:
            params += [servo.servoID]

        self.send(self.CMD_MULT_SERVO_UNLOAD, params)

    def getBatteryVoltage(self):

        self.send(self.CMD_GET_BATTERY_VOLTAGE, [])

        response = self.read()

        if response == None:
            raise TypeError("Response is None")
        
        return (response[1]*256 + response[0]) / 1000.0
    
    def send(self, cmd, params):
        '''
        Constructs a hex packet to send to LeArm through the HID. General format of the hex packets are:

        ???   HEADER  HEADER  LENGTH        Command     PARAMS
        0x0   0x55    0x55    #PARAM + 2    CMD_HEX     HEX ARRAY

        Inputs:
            cmd: One of the command hexs
            params: A a hex array of parameter hexs that are generated from other commands
        '''

        length = len(params) + 2
        hexPacket = [0x0, self.HEADER, self.HEADER, length, cmd] + params
        
        if self.debug:
            print("Sending hex packet: [", end='')
            for HEX in hexPacket:
                print(hex(HEX), end=',')
            print(']')

        self.HIDDevice.write(hexPacket)

    def read(self):

        response = self.HIDDevice.read(64, 50)

        if response != []:

            if self.debug:
                print("Recived response: [", end='')
                for HEX in response:
                    print(hex(HEX), end=',')
                print(']')

            if response[0] == self.HEADER and response[1] == self.HEADER:
                length = response[2]
                return response[4:4 + length]
        
        print("WARNING: Failed to recieve response from arm within 50ms")

    def convertIntToHighLowHex(self, integer):
        low = integer % 256
        high = int((integer - low)/256)

        return [low, high]
    
    def findDuplicates(self, lst):
        duplicateIndices = {}
        
        for index, element in enumerate(lst):
            if element in duplicateIndices:
                duplicateIndices[element].append(index)
            else:
                duplicateIndices[element] = [index]

        return duplicateIndices
