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


def getAllLandmarks(radiographs):
    landmarks = []
    for r in radiographs:
        landmarks += list(r.landMarks.values())
    return landmarks


def getSegmentationFilenames(radiographID):
    segDir = os.path.join(DATA_DIR, "segmentations", "{}-*.png".format(radiographID))

    return glob.glob(segDir)


def getRadiographImage(radiographID):
    """ Returns a tif file from the radiographs directory """
    radioDir = os.path.join(DATA_DIR, "radiographs")

    # Find tif for radiograph number.
    filename = glob.glob(os.path.join(radioDir, "**", "{}.tif".format(radiographID)), recursive=True)[0]

    # Check if the tif of the current radioID is present in our current tifs
    img = Image.open(filename)
