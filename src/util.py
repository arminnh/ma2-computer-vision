import glob
import os

from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "data")

UPPER_TEETH = {1, 2, 3, 4}
LOWER_TEETH = {5, 6, 7, 8}
LEFT_TEETH = {1, 2, 5, 6}
RIGHT_TEETH = {3, 4, 7, 8}
CENTRAL_TEETH = {2, 3, 6, 7}
LATERAL_TEETH = {1, 4, 5, 8}
TOOTH_TYPES1 = {
    "CENTRAL": CENTRAL_TEETH,
    "LATERAL": LATERAL_TEETH,
}
TOOTH_TYPES2 = {
    "UPPER": UPPER_TEETH,
    "LOWER": LOWER_TEETH,
}


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
    img = Image.open(filename)
    return img


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
