import os
import sys
import time
import pickle
import threading

from yaml import safe_load

from datetime import datetime, timezone

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, Angle

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from observational_scheduler.obs_scheduler import target
from logger.astro_logger import astroLogger

from packages.LeArm.Controller.LeArm import LSC_Series_Servo, LeArm

parentPath = os.path.dirname(__file__).replace('/mount', '')

def request_mount_command():
    ### Recieve a command from moxa-pocs/core by loading the pickle instance it has provided in the pickle directory
    with open(f"{parentPath}/pickle/current_target.pickle", "rb") as f:
        current_target = pickle.load(f)

    logger.debug(f"Read current_target.pickle and recieved: {current_target}")

    return current_target

def sendTargetObjectCommand(current_target_object, cmd):
    ### Send a command to other modules via current_target.pickle
    current_target_object.cmd = cmd
    with open(f"{parentPath}/pickle/current_target.pickle", "wb") as f:
        pickle.dump(current_target_object, f)
    logger.debug(f"Sent the following command to current_target.pickle: {cmd}")

def convertAltAztoRaDec(location, az, alt):
    # Az/Alt - astropy input strings in degrees (ex. "90d")
    observationTime = Time(datetime.now(timezone.utc))
    ParkPosLocal = AltAz(az=Angle(az), alt=Angle(alt), location=location, obstime=observationTime)

    return SkyCoord(ParkPosLocal).transform_to('icrs')

def convertRaDectoAltAz(RaDecSkyCoord):
    observationTime = Time(datetime.now(timezone.utc))
    AltAzFrame = AltAz(location=UNIT_LOCATION, obstime=observationTime)
    AltAz = RaDecSkyCoord.transform_to(AltAzFrame)

def connect_to_mount():

    logger.info("Trying to connect to LeArm.")

    LeTelescope = LeArm()
    
    return LeTelescope

def slewToTarget(coordinates, LeMount):
    '''
        Slews to coordinates, from a parked states.

        coordinates is a astropy SkyCoord object

        LeMount is a LeArm instance
    '''

    # Convert coordinates to Alt Az
    AltAz = convertRaDectoAltAz(coordinates)

    # Convert Alt Az to servo degrees
    if AltAz.az.deg > 180: # Using 180 degree servos, so a bearing of 200 degrees coresponds to a bearing of 20 deg on servo6, but pointing servo5 in the opposite direction
        AltAz.az.deg -= 180
        AltAz.alt.deg = 180 - AltAz.alt.deg

    # Move servos
    servos = [LeMount.servo6, LeMount.servo5]
    angles = [AltAz.az.deg, AltAz.alt.deg]
    LeMount.servoMove(servos, angles, time = 1000)

def park(LeMount):
    '''
    Parking behavior to be implemented later, for now look straight up

    LeMount is a LeArm instance.
    '''

    logger.info("Parking the mount.")

    servos = [LeMount.servo6, LeMount.servo5]
    LeMount.servoMove(servos, [90, 90], time = 1000)

    logger.info("Done parking the mount.")

# TODO: LeArm ify
def correctTracking(LeMount, coordinates, astrometryAPI, abortOnFailedSolve):
    '''
     This function is responsible for converting the latest image into a .jpg, uploading it to
     astrometry.net, recieving the plate solved response, and executing a tracking correction 
     on the mount.
     Inputs:
            LeMount
              A LeArm instance.
            
            coordinates:
              An astropy SkyCoord object
            
            astrometryAPI
              An API key specified in settings.yaml, so that an owner of the unit
              can see what images are being plate solved in real time from their 
              dashboard on astrometry.net

            abortOnFailedSolve (not yet implemented)
              A bool that is used to determine what to do after an image comes back
              with a 'failure' status from the astrometry.net API. Unsolvable images
              are of no scientific value to project PANOTPES.
    '''
    if astrometryAPI in (None, "None", 0, False):
        logger.info("Skipping plate solving as directed by the PLATE_SOLVE setting.")
        return
    
    import json
    import random
    import ssl # Need this for the moxa build, consider making into raspbian and moxa packages bc of security risks
    
    from urllib import request, parse

    def getCurrentImageFolder():
        # Find most recent observation directory
        time.sleep(5) # Let camera module make observation folder
        dates = []
        format = "%Y-%m-%d_%H:%M:%S"
        for fileName in os.listdir(f"{parentPath}/images"):
            try:
                dates.append(datetime.strptime(fileName, format))
            except Exception:
                pass

        currentImageFolder = datetime.strftime(max(dates), format)
        logger.debug(f"Found the most recent observation folder: {parentPath}/images/{currentImageFolder}")

    def getNewestImages(previousRawImages):
        # Find the most recent image in the most recent observation folder, search for the first image until a timeout period of 5 minutes
        logger.debug("Waiting for new raw images...")
        timeout = time.time() + 60 * 5
        waitForNewImage = True
        while waitForNewImage:
            rawImages = []
            for dir, subdir, files in os.walk(f"{parentPath}/images/{currentImageFolder}"):
                for file in files:
                    if os.path.splitext(file)[1].lower() in ('.cr2', '.thumb.jpg', '.png'):
                        rawImages.append(os.path.join(dir, file))

            newRawImages = list(set(previousRawImages).symmetric_difference(set(rawImages)))
            previousRawImages = rawImages

            try:
                rawImage = newRawImages[-1]
                logger.debug("Found new raw images.")
                return rawImage, previousRawImages
            except IndexError:
                pass

            if time.time() > timeout:
                logger.warning("Plate-solve timeout reached waiting for new image from camera module. (System is hardcoded to wait 5minutes for an image after calling the camera module)")
                return

            time.sleep(1)

    def plateSolveWithAPI():
        '''
        Wrangles the astrometry API to plate solve the .thumb.jpg. Returns an (RA, DEC) touple
        '''

        logger.debug("Logging into astrometry.net through the API.")
        data = parse.urlencode({'request-json': json.dumps({"apikey": astrometryAPI})}).encode()
        loginRequest = request.Request('http://nova.astrometry.net/api/login', data=data)
        response = json.loads(request.urlopen(loginRequest).read())

        if response['status'] == "success":
            logger.debug("Logged in successfully.")

            session_id = response['session']
            logger.debug(f"Session ID: {session_id}")

            # File uploading, taken from astrometry.net's API documentation and github client
            f = open(rawImage, 'rb')
            file_args = (rawImage, f.read())

            boundary_key = ''.join([random.choice('0123456789') for i in range(19)])
            boundary = '===============%s==' % boundary_key
            headers = {'Content-Type':
                        'multipart/form-data; boundary="%s"' % boundary}
            
            data_pre = (
                '--' + boundary + '\n' +
                'Content-Type: text/plain\r\n' +
                'MIME-Version: 1.0\r\n' +
                'Content-disposition: form-data; name="request-json"\r\n' +
                '\r\n' +
                json.dumps({'session': session_id}) + '\n' +
                '--' + boundary + '\n' +
                'Content-Type: application/octet-stream\r\n' +
                'MIME-Version: 1.0\r\n' +
                'Content-disposition: form-data; name="file"; filename="%s"' % file_args[0] +
                '\r\n' + '\r\n')
            data_post = (
                '\n' + '--' + boundary + '--\n')
            data = data_pre.encode() + file_args[1] + data_post.encode()

            fileUploadRequest = request.Request(url='http://nova.astrometry.net/api/upload', headers=headers, data=data)
            response = json.loads(request.urlopen(fileUploadRequest).read())

            submission_id = response['subid']
            logger.info("Uploaded image to astrometry.net for plate solving.")
            logger.debug(f"submission_id = {submission_id}")

            logger.debug("Waiting for file to start being plate solved...")
            timeout = time.time() + 60 * 5
            waitForQue = True
            while waitForQue:
                time.sleep(1)

                gcontext = ssl.SSLContext() # Again, needed on the moxa system

                jobIDRequest = request.Request(url='https://nova.astrometry.net/api/submissions/' + str(submission_id))
                response = json.loads(request.urlopen(jobIDRequest, context=gcontext).read())
                logger.debug(f"jobIDRequest response: {response}")

                try:
                    jobID = response['jobs'][0]
                    if jobID is not None:
                        logger.debug("Plate solving started.")
                        break
                except IndexError:
                    pass

                if time.time() > timeout:
                    logger.warning("Plate-solve timeout reached waiting for astronomy.net to begin plate solving image.")
                    return

            logger.debug("Waiting for astrometry.net to plate solve...")
            timeout = time.time() + 60 * 10
            waitForPlateSolve = True
            while waitForPlateSolve:
                plateSolveStatusRequest = request.Request(url='https://nova.astrometry.net/api/jobs/' + str(submission_id))
                response = json.loads(request.urlopen(plateSolveStatusRequest, context=gcontext).read())

                if response["status"] == "success":
                    logger.info("Image has been successfully plate solved.")
                    break
                elif response["status"] == "failure":
                    logger.warning("Unable to plate solve image.")
                    return False, False
                elif time.time() > timeout:
                    logger.warning("Plate-solve timeout reached waiting for astronomy.net to plate solve the image.")
                    return
                
                time.sleep(1)
            
            timeout = time.time() + 60 * 10
            waitForImageData = True
            while waitForImageData:
                time.sleep(1)
                    
                imageCoordinatesRequest = request.Request(url='https://nova.astrometry.net/api/jobs/' + str(jobID) + '/calibration')
                response = json.loads(request.urlopen(imageCoordinatesRequest, context=gcontext).read())

                try:
                    RADecimal = response['ra']
                    DECDecimal = response['dec']

                    logger.debug(f"Plate solve coordinates: ra: {RADecimal} dec: {DECDecimal}")
                    return RADecimal, DECDecimal
                except KeyError as e:
                    pass

                if time.time() > timeout:
                    logger.warning("Plate-solve timeout reached waiting for astronomy.net to publish image data.")
                    return

        else:
            logger.warning("Problem logging into the astrometry.net API! Check your API key and internet connection.")

    def executeTrackingCorrection():
        # Perform guiding - aka tracking correction
        # Convert everything to AltAz
        realCords = SkyCoord(RADecimal, DECDecimal, units=(u.deg, u.deg))
        realAltAz = convertRaDectoAltAz(realCords)
        requestedAltAz = convertRaDectoAltAz(coordinates)
        AltCorrection = realAltAz.alt.deg - requestedAltAz.alt.deg
        AzCorrection = realAltAz.az.deg - coordinates.az.deg

        servos = [LeMount.servo6, LeMount.servo5]
        angles = [LeMount.servo6.giveDegrees() + AzCorrection, LeMount.servo5.giveDegrees() + AltCorrection]
        LeMount.servoMove(servos, angles, time=300)

        logger.info("Succesfully executed necessary tracking corrections.")
        time.sleep(15)

    ### Start of correctTracking() ### 
    try:
        currentImageFolder = getCurrentImageFolder()

        previousRawImages = []
        camerasObserving = True
        while camerasObserving:
            rawImage, previousRawImages = getNewestImages(previousRawImages)

            start = time.time() # Time how long it takes to get actual coordinates after taking an image for logging and understanding how plate solve time impacts guiding
            logger.info("Correcting tracking by plate solving...")

            logger.debug(f"Using {rawImage} as latest image.")

            RADecimal, DECDecimal = plateSolveWithAPI()

            if RADecimal is not False: # RADecimal == False if plate solving fails
            
                logger.debug(f"Time spent calculating correction: {time.time() - start}")

                executeTrackingCorrection()
            
            if abortOnFailedSolve and RADecimal == False:
                # TODO: Decide and implement one of the following:
                #          1. Go to the next target
                #          2. Wait X minutes and try again (cloud cover, starlink satelite, whatever)
                #          3. Fully turn off the system
                #          4. Remove this feature and just continue to try and plate solve
                pass

    except Exception as e:
        logger.error("Error during plate solving:", e)

def main():
    global UNIT_LOCATION

    logger.info("Mount module activated.")

    PARENT_DIRECTORY = os.path.dirname(__file__).replace('/mount', '')

    with open(f"{PARENT_DIRECTORY}/conf_files/settings.yaml", 'r') as f:
        settings = safe_load(f)
    logger.debug(f"Read system settings with values: {settings}")

    LAT_CONFIG = settings['LATITUDE']
    LON_CONFIG = settings['LONGITUDE']
    ELEVATION_CONFIG = settings['ELEVATION']
    UNIT_LOCATION = EarthLocation(lat=LAT_CONFIG, lon=LON_CONFIG, height=ELEVATION_CONFIG * u.m)
    ASTROMETRY_API = settings['PLATE_SOLVE']
    ABORT_FAILED_SOLVE_ATTEMPT = settings['ABORT_AFTER_FAILED_SOLVE']

    LeTelescope = connect_to_mount()

    ### Start main mount loop that listens for incoming command from moxa-pocs/core and executes as necessary
    logger.debug("Mount main loop activated.")
    while True:
    
        time.sleep(1)

        current_target = request_mount_command()

        match current_target.cmd:

            case 'slew to target':
                print("System attempting to slew to target...")
                logger.info("Getting ready to slew to target.")

                coordinates = SkyCoord(current_target.position['ra'], current_target.position['dec'], unit=(u.hourangle, u.deg))
                acceptedSlew = slewToTarget(coordinates, LeTelescope)

                sendTargetObjectCommand(current_target, 'take images')
                os.system(f'python3 {parentPath}/cameras/camera_control.py')
                logger.info("Started the camera module.")

                guideThread = threading.Thread(target=correctTracking, args=(LeTelescope, coordinates, ASTROMETRY_API, ABORT_FAILED_SOLVE_ATTEMPT), daemon=True)
                guideThread.start()

            case 'park':
                print("Parking the mount.")
                logger.info("Parking the mount.")
                park(LeTelescope, UNIT_LOCATION)
                sendTargetObjectCommand(current_target, 'parked')
                time.sleep(2)
                break

            case 'observation complete':
                print(f"Done looking at {current_target.name}. Parking the mount.")
                logger.info(f"Done looking at {current_target.name}. Parking the mount.")
                park(LeTelescope, UNIT_LOCATION)
                sendTargetObjectCommand(current_target, 'parked')
                break
            
            case _:
                continue

if __name__ == '__main__':
    logger = astroLogger(enable_color=True)
    main()