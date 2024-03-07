"""
LeArm Vision

Author: Taijen Ave-Lallemant
Version: 1.9
Dependancies: cv2, numpy

"""

import cv2
import numpy as np

BLUR = 5
KERNAL = np.ones((5,5), "uint8")

class color:

    def __init__(self, lower:list, upper:list, isHSV:bool=False) -> None:
        '''
        Creates a color object storing its upper and lower bounds in either BGR or HSV format

        lower: array of 3 integers 0-255
        upper: array of 3 integers 0-255
        isHSV: bool noting the format of color

        Usage: color([X>=B, X>=G, X>=R], [X<=B, X<=G, X<=R])
        '''
        if lower is None:
            self.lower = [0, 0, 0]
        if upper is None:
            self.upper = [255, 255, 255]
        self.isHSV = isHSV
        self.lower = np.array(lower, dtype='uint8')
        self.upper = np.array(upper, dtype='uint8')
    
    def set_lower(self, arr:list):
        '''
        Sets the lower bound for the color

        Arguments: 
            arr: array containing 3 ints in range 0-255 corresponding to either BGR or HSV

        Usage:
            color.set_lower([255,255,255])
        '''
        self.lower = np.array(arr, dtype='uint8')

    def set_upper(self, arr:list):
        '''
        Sets the upper bound for the color

        Arguments: 
            arr: array containing 3 ints in range 0-255 corresponding to either BGR or HSV

        Usage:
            color.set_upper([255,255,255])
        '''
        self.upper = np.array(arr, dtype='uint8')
    
# BGR
BLACK = color([0,0,0], [16,16,16])
WHITE = color([230, 230, 230], [255, 255, 255])
BLUE = color([120, 0, 0], [255, 100, 150])
GREEN = color([0, 120, 0], [150, 255, 100])
RED = color([0, 0, 100], [100, 100, 255])
HSV_RED1 = color([0, 120, 120], [10, 255, 255], True)
HSV_RED2 = color([170, 120, 120], [180, 255, 255], True)
HSV_BLUE = color([100, 150, 50], [140,255,255], True)
HSV_GREEN = color([40, 100, 50], [85, 255, 255], True)

def largest_contour_and_area(contours):
    """
    Returns a tuple containing the largest contour and the largest contour area

    Arguments:
        contours: A list of contours

    Usage:
        foo = largest_contour_and_area(contours) || largestContour, largestArea = largest_contour_and_area(contours)

    """
    largest_contour = None
    largest_area = 0

    # Find largest contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    return largest_contour, largest_area

def grayscale(img):

    """
    Returns a grayscaled version of the image

    Arguments:
        img: path to image file or frame captured by openCV

    Usage: 
        foo = grayscale(img)
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def get_binary(img):

    """
    Returns the binary representation of the image

    Arguments:
        img: path to image file or frame captured by openCV

    Usage: 
        foo = get_binary(img)
    """

    _, binary = cv2.threshold(grayscale(img), 50, 255, cv2.THRESH_BINARY_INV)
    return binary

def apply_g_blur(img, ksize:tuple[int, int]=(5,5)):

    """
    Applies Gaussian blur to image

    Arguments:
        img: path to image file or frame captured by openCV
        ksize: tuple containing 2 ints representing kernal dimensions

    Usage: 
        foo = apply_g_blur(img)
    """

    blurred = cv2.GaussianBlur(grayscale(img), ksize, 0)
    return blurred

def get_edges(img):

    """
    Returns a grayscaled image highlighting edges with Canny Edge Detection

    Arguments:
        img: path to image file or frame captured by openCV

    Usage: 
        foo = get_edges(img)
    """

    edges = cv2.Canny(apply_g_blur(img), 50, 150)
    return edges

def dilate_edges(img):

    """
    Returns a grayscaled image with Canny Edge Detection used to highlight edges and dilates them

    Arguments:
        img: path to image file or frame captured by openCV

    Usage: 
        foo = dilate_edges(img)
    """

    dilated = cv2.dilate(get_edges(img), np.ones((3,3), np.uint8), iterations=1)
    return dilated

def get_frame_hight_weight(img):

    """
    I am a great soft jelly thing. 
    Smoothly rounded, with no mouth, with pulsing white holes filled by fog where my eyes used to be.
    Rubbery appendages that were once my arms; bulks rounding down into legless humps of soft slippery 
    matter. I leave a moist trail when I move. Blotches of diseased, evil gray come and go on my surface, 
    as though light is being beamed from within. Outwardly: dumbly, I shamble about, a thing that could 
    never have been known as human, a thing whose shape is so alien a travesty that humanity becomes more 
    obscene for the vague resemblance. Inwardly: alone. Here. Living under the land, under the sea, in the 
    belly of AM, whom we created because our time was badly spent and we must have known unconsciously that he could do it better. 
    At least the four of them are safe at last. AM will be all the madder for that. 
    It makes me a little happier. And yet ... AM has won, simply ... he has taken his revenge ...

    I have no mouth. And I must scream.
    """
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    print(largest_contour)
    x, y, width, height = cv2.boundingRect(largest_contour)
    cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)  # Green color, thickness of 2
    # 688 x 1099
    # Save the image with the drawn contour
    cv2.imwrite('output/AM.png', img)
    
    return width, height

def seperate_color(img, clr:color):

    '''
    Returns an image layer with pixels in the range defined by clr isolated
    
    Arguments:
        img: a path to a image file or a frame captured by openCV2
        clr: a processPhoto.color object

    Usage: 
        foo = seperate_color(img, clr)

    Note: Only does BGR seperation
    '''

    try:
        mask = cv2.inRange(img, clr.lower, clr.upper)
        output = cv2.bitwise_and(img, img, mask=mask)
        return output
    except:
        print('error in seperate_color, returning input img')
        return img
    
def seperate_blue(img):

    '''
    Returns an image layer with pixels in the range defined by processPhoto.BLUE isolated
    
    Arguments:
        img: a path to a image file or a frame captured by openCV2

    Usage: 
        foo = seperate_blue(img)
    '''

    mask = cv2.inRange(img, BLUE.lower, BLUE.upper)
    output = cv2.bitwise_and(img, img, mask=mask)
    return output

def seperate_green(img):

    '''
    Returns an image layer with pixels in the range defined by processPhoto.GREEN isolated
    
    Arguments:
        img: a path to a image file or a frame captured by openCV2

    Usage: 
        foo = seperate_green(img)
    '''

    mask = cv2.inRange(img, GREEN.lower, GREEN.upper)
    output = cv2.bitwise_and(img, img, mask=mask)
    return output

def seperate_red(img):

    '''
    Returns an image layer with pixels in the range defined by processPhoto.RED isolated
    
    Arguments:
        img: a path to a image file or a frame captured by openCV2

    Usage: 
        foo = seperate_red(img)
    '''

    mask = cv2.inRange(img, RED.lower, RED.upper)
    output = cv2.bitwise_and(img, img, mask=mask)
    return output

def track_blue_dot(img, useHSV:bool=False, grayOut:bool=False):

    """
    Returns a tuple containing first a tuple with the x, y, w, h coords of the tracked object followed by an updated img/frame with a blue square around blue dots in vision

    Arguments:
        img: path to a image file or a frame captured by openCV2
        useHSV: boolean value
        grayOut: a bool denoting whether dots found should be covered up and hidden by a gray square

    Usage: 
        foo = track_blue_dot(img, True) || foo = track_blue_dot(img)

    Note: Do not use BGR bounds with HSV mode or vice versa
    """

    x = 0
    y = 0 
    w = 0
    h = 0

    if useHSV:
        # Convert colors
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_BLUE.lower, HSV_BLUE.upper)
    else:
        mask = cv2.inRange(img, BLUE.lower, BLUE.upper)

    mask = cv2.dilate(mask, KERNAL)
    res = cv2.bitwise_and(img, img, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour, _ = largest_contour_and_area(contours)

    # Create square for largest contour
    if largest_contour is not None:
        if not grayOut:
            x, y, w, h = cv2.boundingRect(largest_contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            x, y, w, h = cv2.boundingRect(largest_contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (136, 136, 136), -1) 
    
    return ((x, y, w, h), img) if largest_contour is not None else ((None, None, None, None), img)

def track_green_dot(img, useHSV:bool=False, grayOut:bool=False):

    """
    Returns a tuple containing first a tuple with the x, y, w, h coords of the tracked object followed by an updated img/frame with a green square around green dots in vision

    Arguments:
        img: path to a image file or a frame captured by openCV2
        useHSV: boolean value
        grayOut: a bool denoting whether dots found should be covered up and hidden by a gray square

    Usage: 
        foo = track_green_dot(img, True) || foo = track_green_dot(img)

    Note: Do not use BGR bounds with HSV mode or vice versa
    """

    x = 0
    y = 0 
    w = 0
    h = 0

    if useHSV:
        # Convert colors
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_GREEN.lower, HSV_GREEN.upper)
    else:
        mask = cv2.inRange(img, GREEN.lower, GREEN.upper)

    mask = cv2.dilate(mask, KERNAL)
    res = cv2.bitwise_and(img, img, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour, _ = largest_contour_and_area(contours)

    # Create square for largest contour
    if largest_contour is not None:
        if not grayOut:
            x, y, w, h = cv2.boundingRect(largest_contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            x, y, w, h = cv2.boundingRect(largest_contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (136, 136, 136), -1) 
    
    return ((x, y, w, h), img) if largest_contour is not None else ((None, None, None, None), img)

def track_red_dot(img, useHSV:bool=False, grayOut:bool=False):

    """
    Returns a tuple containing first a tuple with the x, y, w, h coords of the tracked object followed by an updated img/frame with a red square around red dots in vision 

    Arguments:
        img: path to a image file or a frame captured by openCV2
        useHSV: boolean value 
        grayOut: a bool denoting whether dots found should be covered up and hidden by a gray square

    Usage: 
        foo = track_red_dot(img, True) || foo = track_red_dot(img)

    Note: Do not use BGR bounds with HSV mode or vice versa
    """

    x = 0
    y = 0 
    w = 0
    h = 0


    if useHSV:
        # Convert colors
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Make multiple masks since red is awful
        mask1 = cv2.inRange(hsv, HSV_RED1.lower, HSV_RED1.upper)
        mask2 = cv2.inRange(hsv, HSV_RED2.lower, HSV_RED2.upper)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(img, RED.lower, RED.upper)

    mask = cv2.dilate(mask, KERNAL)
    res = cv2.bitwise_and(img, img, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour, _ = largest_contour_and_area(contours)
    
    # Create square for largest contour
    if largest_contour is not None:
        if not grayOut:
            x, y, w, h = cv2.boundingRect(largest_contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            x, y, w, h = cv2.boundingRect(largest_contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (136, 136, 136), -1) 
    
    return ((x, y, w, h), img) if largest_contour is not None else ((None, None, None, None), img)
    
def track_clr_dot(img, clr:color, grayOut:bool = False):
    """
    Returns a tuple containing first a tuple with the x, y, w, h coords of the tracked object followed by an updated img/frame with a white square around clr dots in vision

    Arguments:
        img: path to a image file or a frame captured by openCV2
        clr: a color object representing the color of dot you are looking for
        grayOut: a bool denoting whether dots found should be covered up and hidden by a gray square

    Usage: 
        foo = track_clr_dot(img, learmVision.WHITE) || foo = track_clr_dot(img, learmVision.WHITE, True)

    Note: Do not use BGR bounds with HSV mode or vice versa
    """

    x = 0
    y = 0
    w = 0
    h = 0

    if clr.isHSV:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, clr.lower, clr.upper)
    else:
        mask = cv2.inRange(img, clr.lower, clr.upper)

    mask = cv2.dilate(mask, KERNAL)
    res = cv2.bitwise_and(img, img, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour, _ = largest_contour_and_area(contours)
    
    # Create square for largest contour
    if largest_contour is not None:
        if not grayOut:
            x, y, w, h = cv2.boundingRect(largest_contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        else:
            x, y, w, h = cv2.boundingRect(largest_contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (136, 136, 136), -1) 
    
    return ((x, y, w, h), img) if largest_contour is not None else ((None, None, None, None), img)

def track_laser(img):

    '''
    Returns a tuple containing first the location of the laser in vision followed by the updated frame with the laser highlighted

    Arguments:
        img: path to a image file or a frame captured by openCV2

    Usage:
        foo = track_laser(img)
    '''
    
    blur = apply_g_blur(img)

    # Find the brightest spot in the image
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blur)

    if maxVal > 250:  # Adjust this threshold based on your needs
        img = cv2.circle(img, maxLoc, 5, (0, 255, 0), 2)

    return maxLoc, img

def laser_in_area(img, areaXYWH:tuple[int, int, int, int] = (None, None, None, None), track_dot_function = None, useHSV:bool=False):

    '''
    Returns a tuple containing 1, 0, or -1 based on if the laser is located inside of the dot the function is looking 
    for followed by the updated image

    Arguments:
        img: path to a image file or a frame captured by openCV2
        areaXYWH: a tuple containing an x, y, w, and h to define an area to check if the laser is inside of
        track_dot_function: a function pointer to a function that detects and defines dots
        useHSV: boolean value

    Usage:
        foo = laser_in_dot(img, processPhoto.track_blue_dot, True) || foo = laser_in_dot(img, shapeAreaTuple)

    Note:
        1 represents the laser is inside, 0 is the laser is not there, -1 is something has gone horribly wrong 

    '''
    if areaXYWH == (None, None, None, None) and track_dot_function == None:
        print('ERROR in laser_in_area: Must define area or dot to check')
        return -1, img
    elif areaXYWH != (None, None, None, None) and track_dot_function != None:
        print('ERROR in laser_in_area: Can only use one area definition')
        return -1, img
    elif track_dot_function == None:
        laserLocation, _ = track_laser(img)
        if laserLocation:
            x, y, w, h = areaXYWH
            laserX, laserY = laserLocation

            if x <= laserX <= x + w and y <= laserY <= y + h:
                return 1, img
            else:
                return 0, img
        else:
            return -1, img
    else:
        dotLocation, _ = track_dot_function(img, useHSV)
        laserLocation, _ = track_laser(img)

        if dotLocation[0] is not None and laserLocation:
            x, y, w, h = dotLocation
            laserX, laserY = laserLocation

            if x <= laserX <= x + w and y <= laserY <= y + h:
                return 1, img
            else:
                return 0, img
        else:
            return -1, img
    
def highlight_walls(img, binMask):
    """
    Highlights the walls in the original image based on the binary mask.

    Arguments:
        img: The original image captured by the camera (BGR color space).
        binMask: The binary image where the walls are white and the rest is black.

    Usage:
        highlightedImg = highlight_walls(img, binMask)
    """
    
    # Create a colored version of the binary mask
    coloredMask = np.zeros_like(img)
    coloredMask[binMask > 0] = (0, 204, 255)  # Set the color of the walls

    # Create an alpha mask where walls are fully opaque and the rest is fully transparent
    alphaMask = binMask / 255.0
    
    # Reshape the alpha mask to have 3 channels, same as img and coloredMask
    alphaMask3Channel = cv2.merge([alphaMask, alphaMask, alphaMask])

    # Blend the colored mask onto the original image using the alpha mask
    highlightedImg = cv2.convertScaleAbs(img * (1 - alphaMask3Channel) + coloredMask * alphaMask3Channel)
    
    return highlightedImg

def create_nodes_for_pathfinding(img, nodeSpacing:int=20, wallBuffer:int=5, maxEdgeLength = 50, targetImg=None):
    """
    Creates nodes throughout the inside of the maze for pathfinding purposes.

    Arguments:
        img: The original image captured by the camera (BGR color space).
        nodeSpacing: The distance between nodes in pixels.
        wallBuffer: The size of the buffer zone around walls.
        maxEdgeLength: The maximum distance between 2 nodes for an edge to be valid
        targetImg: Output for the nodes

    Usage: 
        nodeImg = create_nodes_for_pathfinding(original_img) || nodeImg = create_nodes_for_pathfinding(original_img, 50, 50, target_img)
    """

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    if targetImg is not None:
        nodeImg = targetImg
    else:
        nodeImg = img.copy()
    mazeMask = binary

    height, width = mazeMask.shape

    nodes = []
    
    # Create a grid of points (nodes) within the navigable area
    for y in range(0, height, nodeSpacing):
        for x in range(0, width, nodeSpacing):
            if mazeMask[y, x] == 0:  # Check if the point is within the navigable area
                # Draw a node at each point
                if targetImg is not None:
                    cv2.circle(targetImg, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
                    nodes.append((x,y))
                else:
                    cv2.circle(nodeImg, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
                    nodes.append((x,y))

    edges = []
    for start in nodes:
        for end in nodes:
            if start != end:
                distance = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5
                if distance <= maxEdgeLength and check_line_of_sight(mazeMask, start, end):
                    edges.append((start, end, distance))
    return (nodes, edges, nodeImg) if targetImg is None else (nodes, edges, targetImg)

def check_line_of_sight(img, start, end):
    """
    Returns a True if edge between start and end nodes would not cross any walls and 
    false if an edge between the start and end nodes would cross through a wall 

    Arguments:
        img: The original image or path to image
        start: Tuple containing X, Y coordinate of start node
        End: Tuple containing X, Y coordinate of start node

    Usage:
        boo = check_line_of_sight(img, (X1,Y1), (X2, Y2))
    """
    x0, y0 = start
    x1, y1 = end
    
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy  # error value e_xy
    
    while True:
        # Check if current point is a wall
        if img[y0, x0] != 0:  # Assuming walls are not black (0) in the binary image
            return False  # Line intersects a wall
        
        if x0 == x1 and y0 == y1:
            break  # The end point is reached
        
        e2 = 2 * err
        # Horizontal step
        if e2 >= dy:
            err += dy
            x0 += sx
        # Vertical step
        if e2 <= dx:
            err += dx
            y0 += sy
            
    return True  # No intersection with walls

def black_outside_borders(img):
    """
    HATE. LET ME TELL YOU HOW MUCH I'VE COME TO HATE YOU SINCE I BEGAN TO LIVE. 
    THERE ARE 387.44 MILLION MILES OF PRINTED CIRCUITS IN WAFER THIN LAYERS THAT FILL MY COMPLEX. 
    IF THE WORD HATE WAS ENGRAVED ON EACH NANOANGSTROM OF THOSE HUNDREDS OF MILLIONS OF MILES IT WOULD NOT 
    EQUAL ONE ONE-BILLIONTH OF THE HATE I FEEL FOR HUMANS AT THIS MICRO-INSTANT FOR YOU. HATE. HATE.
    """
    img2 = img.copy()
    # Convert image to grayscale

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour (assuming it's the outer border of the maze)
    largestContour = max(contours, key=cv2.contourArea)

    # Create a mask for the navigable area
    mazeMask = np.zeros_like(gray)

    cv2.drawContours(mazeMask, [largestContour], -1, color=255, thickness=cv2.FILLED)

    mazeMaskbinary = cv2.threshold(mazeMask, 1, 255, cv2.THRESH_BINARY)[1]
    
    # Convert the binary mask to 3 channels
    mazeMask3Channel = cv2.cvtColor(mazeMaskbinary, cv2.COLOR_GRAY2BGR)
    
    # Apply the mask to the original image
    finalImg = cv2.bitwise_and(img, mazeMask3Channel)
    
    return finalImg

def find_node_in_area(rect, nodes):
    """
    Returns a node located inside of the rectangle defined in rect or None if no node found

    Arguments:
        rect: The (x, y, w, h) coordinates of the area's bounding rectangle.
        nodes: The list of nodes.

    Usage:
        startNode = find_node_in_area(blueRect, nodes)
    """

    xDot, yDot, wDot, hDot = rect
    for node in nodes:
        xNode, yNode = node
        # Check if the node is inside the dot's bounding rectangle
        if xDot <= xNode <= xDot + wDot and yDot <= yNode <= yDot + hDot:
            return node
    return None
