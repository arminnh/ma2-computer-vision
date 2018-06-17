import glob
import math
import os
import time

import numpy as np
import scipy.interpolate


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "data")

RADIOGRAPH_NUMBERS = list(range(1, 15))
TEETH = {1, 2, 3, 4, 5, 6, 7, 8}
UPPER_TEETH = {1, 2, 3, 4}
LOWER_TEETH = {5, 6, 7, 8}
LEFT_TEETH = {1, 2, 5, 6}
RIGHT_TEETH = {3, 4, 7, 8}
CENTRAL_TEETH = {2, 3, 6, 7}
LATERAL_TEETH = {1, 4, 5, 8}

SAMPLE_AMOUNT = 25

PCA_COMPONENTS = 20

np.seterr(all='raise')


class Timer:
    def __init__(self, message):
        self.message = message
        self.start = 0
        print("> {}".format(self.message))

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(">> {:.2f} seconds".format(time.time() - self.start))


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
    Returns points on a line that goes through `current` with slope m.
    :param m: the slope of the line to sample on
    :param current: the position on the line to sample on each side from
    :param pixelsToSample: The amount of pixels on the line to sample on each side of m
    """
    x = current[0]
    b = current[1]

    # dx = the distance that can be moved in X for there to be a max distance of 1 in Y
    dx = 1 / m if m != 0 else 1

    xOffset = pixelsToSample * dx if -1 < dx < 1 else pixelsToSample
    pStart = (x - xOffset)
    pEnd = (x + xOffset)

    # 2 * pixelsToSample + 1 => as in "we have 2k+1 samples which can be put into a vector
    X = np.linspace(pStart, pEnd, pixelsToSample+1)
    Y = m * (X - current[0]) + b

    X = X.round().astype(int)
    Y = Y.round().astype(int)

    return list(zip(X, Y))


def getNormalSlope(before, current, nextt):
    xx = np.asarray([before[0], current[0], nextt[0]])
    yy = np.asarray([before[1], current[1], nextt[1]])
    sorted_xx = xx.argsort()

    # For scipy strictly increasing x values
    xx = xx[sorted_xx] + [0, 0.00000001, 0.00000002]
    yy = yy[sorted_xx]

    f = scipy.interpolate.CubicSpline(xx, yy).derivative()
    tangentLineSlope = f(current[0])

    # m = slope of normal line
    m = -1 / tangentLineSlope if tangentLineSlope != 0 else 0
    return m

def getCentersOfInitModel(landmark):
    points = landmark.getPointsAsTuples()

    splitted = [list(points[40*i:40*(i+1)]) for i in range(4)]
    return np.mean(splitted,1)

