import glob
import math
import os

import numpy as np
import scipy.interpolate
from PIL import Image

from preprocess_img import PILtoCV, applyCLAHE, cvToPIL, bilateralFilter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "data")

RADIOGRAPH_NUMBERS = list(range(1, 15))
TEETH = {1, 2, 3, 4, 5, 6, 7, 8}
UPPER_TEETH = {1, 2, 3, 4}
LOWER_TEETH = {5, 6, 7, 8}
LEFT_TEETH = {1, 2, 5, 6}
RIGHT_TEETH = {3, 4, 7, 8}
CENTRAL_TEETH = {2, 3, 6, 7}
LATERAL_TEETH = {1, 4, 5, 8}

SAMPLE_AMOUNT = 40  # 10, 13, 15, 30
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


def loadRadiographImage(radiographFilename):
    """ Returns a tif file from the radiographs directory """
    radioDir = os.path.join(DATA_DIR, "radiographs")

    # Find tif for radiograph number.
    filename = glob.glob(os.path.join(radioDir, "**", "{}.tif".format(radiographFilename)), recursive=True)[0]

    # Check if the tif of the current radioID is present in our current tifs
    return Image.open(filename).convert("L")


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


def sampleNormalLine(m, current, pixelsToSample):
    """
    Returns points on the normal line that goes through `current` by calculating the angle between `before` and `next`.
    :param pixelsToSample: If given, samples a certain amount of pixels on each side of `current`
    """
    x = current[0]
    b = current[1]
    # dx = the distance that can be moved in X for there to be a max distance of 1 in Y
    dx = 1 / m if m != 0 else 1

    # pixels = 30
    xOffset = pixelsToSample * dx if -1 < dx < 1 else pixelsToSample
    pStart = (x - xOffset)
    pEnd = (x + xOffset)
    X = np.asarray([pStart + k / (pixelsToSample - 1) * (pEnd - pStart) for k in range(pixelsToSample)])
    Y = m * (X - current[0]) + b

    # x1, y1 = X[0], Y[0]
    # x2, y2 = X[-1], Y[-1]
    # print("\nm: {}, dx: {}, line length: {}".format(m, dx, math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)))

    # if pixelsToSample is not None:
    #     # TODO: sample in a better way
    #
    #     beforeCurrentIdx = np.random.randint(0, round(len(X) / 2), pixelsToSample)
    #     afterCurrentIdx = np.random.randint(round(len(X) / 2), len(X), pixelsToSample)
    #     beforeCurrentIdx.sort()
    #     afterCurrentIdx.sort()
    #     X = np.concatenate((X[beforeCurrentIdx], X[afterCurrentIdx]))
    #     Y = np.concatenate((Y[beforeCurrentIdx], Y[afterCurrentIdx]))

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
