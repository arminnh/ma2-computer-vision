import glob
import os

import cv2
import numpy as np
import scipy.interpolate

from preprocess_img import bilateralFilter, applyCLAHE
from util import DATA_DIR


def getPixelProfile(img, points, derive=True):
    # Get pixel values on the given points
    pixels = np.asarray([img[y, x] for (x, y) in points], dtype=np.int)

    rawPixelProfile = pixels.copy()
    if not derive:
        return rawPixelProfile, None, None

    # Derivative profile of length n_p - 1
    derivedProfile = np.asarray([pixels[i + 1] - pixels[i - 1] for i in range(1, len(pixels) - 1)])

    # Normalized derivative profile
    scale = np.sum(np.abs(pixels))
    if scale != 0:
        normalizedProfile = pixels / scale
    else:
        normalizedProfile = pixels.copy()

    return rawPixelProfile, derivedProfile, normalizedProfile


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
    _, xMax = img.shape
    yMax = yMax - yMin

    pathX = np.linspace(0, xMax - 1, xMax / 20).astype(int)
    pathY = np.zeros(len(pathX)).astype(int)

    # trellis (x, y, 2) shape. 2 to hold cost and previousY
    trellis = np.full((len(pathX), yMax, 2), np.inf)

    # set first column in trellis
    for y in range(yMax):
        trellis[0, y, 0] = img[y + yMin, 0]
        trellis[0, y, 1] = y

    # forward pass
    for i in range(1, len(pathX)):
        x = pathX[i]

        for y in range(yMax):
            # yWindow the area to look in for the next points
            yWindow = 10
            start = y - yWindow if y > yWindow - 1 else y
            end = y + yWindow if y < yMax - yWindow else y

            bestPrevY = trellis[i - 1, start:end + 1, 0].argmin() + y - yWindow
            bestPrevCost = trellis[i - 1, bestPrevY, 0]

            # new cost = previous best cost + current cost (colour intensity)
            trellis[i, y, 0] = bestPrevCost + img[y + yMin, x]  # + self.findLineForJawSplitTransitionCost(bestPrevY, y)
            trellis[i, y, 1] = bestPrevY

    # find the best path, backwards pass
    # set first previousY value to set up backwards pass
    previousY = int(trellis[-1, :, 0].argmin())

    for i in range(len(pathX) - 1, -1, -1):
        pathY[i] = previousY + yMin
        previousY = int(trellis[i, previousY, 1])

    return np.asarray(list(zip(pathX, pathY)))


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

    # Open the radiograph image in grayscale.
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Select a region of interest
    img, XOffset, YOffset = cropRegionOfInterest(img)

    # preprocess the image
    img = preprocessRadiographImage(img)

    # Split image in two: upper and lower jaw
    # Find line to split jaws into two images. Only search in a certain y range.
    yMax, xMax = img.shape
    ySearchMin, ySearchMax = int((yMax / 2) - 200), int((yMax / 2) + 300)

    imgUpperJaw, imgLowerJaw = img.copy(), img.copy()
    jawSplitLine = findLineForJawSplit(img, ySearchMin, ySearchMax)
    # interpF = scipy.interpolate.CubicSpline(jawSplitLine[:, 0], jawSplitLine[:, 1])
    #
    # for x in range(xMax):
    #     y = int(interpF(x))
    #     imgUpperJaw[y + 1:-1, x] = 255
    #     imgLowerJaw[0:y, x] = 255

    return img, imgUpperJaw, imgLowerJaw, jawSplitLine, XOffset, YOffset
