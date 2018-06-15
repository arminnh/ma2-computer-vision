import glob
import math
import os
import time

import numpy as np
import scipy.interpolate
from PIL import Image

from preprocess_img import bilateralFilter, applyCLAHE

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "data")

RADIOGRAPH_NUMBERS = list(range(1, 15))
TEETH = {1, 2, 3, 4, 5, 6, 7, 8}
UPPER_TEETH = {1, 2, 3, 4}
LOWER_TEETH = {5, 6, 7, 8}
LEFT_TEETH = {1, 2, 5, 6}
RIGHT_TEETH = {3, 4, 7, 8}
CENTRAL_TEETH = {2, 3, 6, 7}
LATERAL_TEETH = {1, 4, 5, 8}

SAMPLE_AMOUNT = 20  # 10, 13, 15, 30

PCA_COMPONENTS = 20


def getRadiographFilenames(number=None, extra=False):
    radiographDir = os.path.join(DATA_DIR, "radiographs")

    if extra:
        radiographDir = os.path.join(radiographDir, "**")

    if number is None:
        radiographDir = os.path.join(radiographDir, "*.tif")
    else:
        radiographDir = os.path.join(radiographDir, "{}.tif".format(number))

    return glob.glob(radiographDir, recursive=True)


def getLandmarkFilenames(radiographFilename):
    landmarkDir = os.path.join(DATA_DIR, "landmarks", "**", "landmarks{}-*.txt".format(radiographFilename))

    return glob.glob(landmarkDir, recursive=True)


def getSegmentationFilenames(radiographFilename):
    segDir = os.path.join(DATA_DIR, "segmentations", "{}-*.png".format(radiographFilename))

    return glob.glob(segDir)


def preprocessRadiographImage(image, transformations=None):
    if transformations is None:
        transformations = [
            bilateralFilter,
            applyCLAHE,
            bilateralFilter,
            applyCLAHE,
        ]

    for transform in transformations:
        image = transform(image)

    return image


def findLineForJawSplit(img, yMin, yMax):
    """
    Find the best path to split the jaws in the image starting from position (0, y).
    The path consists of y values, the indices are x values.
    :type img: np.ndarray
    """
    print("findLineForJawSplit")
    tttime = time.time()
    ttime = time.time()
    _, xMax = img.shape
    yMax = yMax - yMin

    # trellis (y, x, 2) shape. 2 to hold cost and previousY
    trellis = np.full((yMax, xMax, 2), np.inf)

    print("init trellis", time.time() - ttime)
    ttime = time.time()

    # set first column in trellis
    for y in range(yMax):
        trellis[y, 0, 0] = img[y + yMin, 0]
        trellis[y, 0, 1] = y

    print("init first y trellis", time.time() - ttime)
    ttime = time.time()

    # forward pass
    for x in range(1, xMax):
        for y in range(yMax):
            start = y - 1 if y > 0 else y
            end = y + 2 if y < yMax - 1 else y

            bestPrevY = trellis[start:end, x - 1, 0].argmin() + y - 1
            bestPrevCost = trellis[bestPrevY, x - 1, 0]

            # new cost = previous best cost + current cost (colour intensity)
            trellis[y, x, 0] = bestPrevCost + img[y + yMin, x]  # + self.transitionCost(bestPrevY, y)
            trellis[y, x, 1] = bestPrevY

    print("forward pass", time.time() - ttime)
    ttime = time.time()

    # find the best path, backwards pass
    path = []

    # set first previousY value to set up backwards pass
    previousY = int(trellis[:, -1, 0].argmin())

    for x in range(xMax - 1, -1, -1):
        path.insert(0, previousY + yMin)
        previousY = int(trellis[previousY, x, 1])

    print("find path", time.time() - ttime)
    ttime = time.time()

    print("total", time.time() - tttime)
    raise ValueError
    return path


def findLineForJawSplitTransitionCost(prevY, currY):
    if prevY == currY:
        return 2
    elif currY - prevY == 1 or currY - prevY == -1:
        return 1
    return np.inf


def cropRegionOfInterest(img):
    y, x = img.shape
    xMid, yMid = int(x / 2), int(y / 2)
    xStart, yStart = 375, 450

    cropY = slice(yMid - yStart, yMid + yStart + 250)
    cropX = slice(xMid - xStart, xMid + xStart)

    # Cut out the region of interest
    img = img[cropY, cropX]

    # X and Y offsets for the landmark points based on the region of interest selected
    XOffset = - (xMid - xStart)
    YOffset = - (yMid - yStart)

    return img, XOffset, YOffset


def loadRadiographImage(radiographFilename):
    """ Returns a tif file from the radiographs directory """
    radioDir = os.path.join(DATA_DIR, "radiographs")

    # Find tif for radiograph number.
    filename = glob.glob(os.path.join(radioDir, "**", "{}.tif".format(radiographFilename)), recursive=True)[0]

    # Open the radiograph image, convert it to grayscale, and convert the PIL Image to an np array
    img = np.array(Image.open(filename).convert("L"))

    # Select a region of interest
    img, XOffset, YOffset = cropRegionOfInterest(img)

    # preprocess the image
    img = preprocessRadiographImage(img)

    # Split image in two: upper and lower jaw
    # Find line to split jaws into two images. Only search in a certain y range.
    yMax, _ = img.shape
    ySearchMin, ySearchMax = int((yMax / 2) - 200), int((yMax / 2) + 300)
    jawSplitLine = findLineForJawSplit(img, 400, 675)

    imgUpperJaw, imgLowerJaw = img.copy(), img.copy()
    for x, y in enumerate(jawSplitLine):
        imgUpperJaw[y + 1:-1, x] = 255
        imgLowerJaw[0:y, x] = 255

    return img, imgUpperJaw, imgLowerJaw, jawSplitLine, XOffset, YOffset


def flipToothNumber(n):
    if n == 1:
        return 4
    elif n == 2:
        return 3
    elif n == 3:
        return 2
    elif n == 4:
        return 1
    elif n == 5:
        return 8
    elif n == 6:
        return 7
    elif n == 7:
        return 6
    elif n == 8:
        return 5


def getSlope(p1, p2):
    """ Returns the slope between the two given points. """
    (x1, y1) = p1
    (x2, y2) = p2
    if x1 == x2:
        return 99999999
    return (y2 - y1) / (x2 - x1)


def rotateSlope(m, theta):
    """ Rotates the given slope for a certain angle. """
    return (math.sin(theta) + m * math.cos(theta)) / (math.cos(theta) - m * math.sin(theta))


def getSlopeOfInnerBisector(m1, m2):
    """ Returns the slope of the bisector that lies inside the two lines defined by slopes m1 and m2. """
    # calculate angle between the two outer points
    theta = math.atan(abs((m1 - m2) / (1 + m1 * m2)))

    m3 = rotateSlope(m2, theta / 2)
    m4 = rotateSlope(m2, -theta / 2)

    if round(math.atan(abs((m1 - m3) / (1 + m1 * m3))), 6) == round(theta / 2, 6):
        return -1 / m3
    elif round(math.atan(abs((m1 - m4) / (1 + m1 * m4))), 6) == round(theta / 2, 6):
        return -1 / m4


def sampleLine(m, current, pixelsToSample):
    """
    Returns points on the normal line that goes through `current` by calculating the angle between `before` and `next`.
    :param m: the slope of the line to sample on
    :param current:  the position on the line to sample on each side from
    :param pixelsToSample: If given, samples a certain amount of pixels on each side of `current`
    """
    x = current[0]
    b = current[1]

    # dx = the distance that can be moved in X for there to be a max distance of 1 in Y
    dx = 1 / m if m != 0 else 1

    xOffset = pixelsToSample * dx if -1 < dx < 1 else pixelsToSample
    pStart = (x - xOffset)
    pEnd = (x + xOffset)

    # 2 * pixelsToSample + 1 => as in "we have 2k+1 samples which can be put into a vector
    X = np.linspace(pStart, pEnd, pixelsToSample + 1)
    Y = m * (X - current[0]) + b

    return list(zip(X, Y))


def getNormalSlope(before, current, nextt):
    xx = np.asarray([before[0], current[0], nextt[0]])
    yy = np.asarray([before[1], current[1], nextt[1]])
    sorted_xx = xx.argsort()
    # Fuck you scipy and your strictly increasing x values
    xx = xx[sorted_xx] + [0, 0.00000001, 0.00000002]
    yy = yy[sorted_xx]
    # y = m x + b
    f = scipy.interpolate.CubicSpline(xx, yy).derivative()
    tangentLineSlope = f(current[0])
    # m = slope of normal line
    m = -1 / tangentLineSlope if tangentLineSlope != 0 else 0
    return m


def getPixels(radiograph, points, getDeriv=True):
    # Get pixel values on the sampled positions
    img = radiograph.image  # type: Image
    pixels = np.asarray([img[int(y), int(x)] for (x, y) in points])

    beforeDeriv = pixels.copy()
    if getDeriv:
        # Derivative profile of length n_p - 1
        pixels = np.asarray([pixels[i + 1] - pixels[i - 1] for i in range(len(pixels) - 1)])  # np.diff(pixels)

    afterDeriv = pixels.copy()

    # Normalized derivative profile
    # print("i {}, derivated profile: {}, divisor: {}".format(i, list(pixels), np.sum(np.abs(pixels))), end=", ")
    scale = np.sum(np.abs(pixels))
    if scale != 0:
        pixels = pixels / scale

    scaled = pixels.copy()

    return beforeDeriv, afterDeriv, scaled
