import os
import re

from PIL import Image

from helpers import getSegmentationFilenames


class Segment:
    def __init__(self, radiographID, toothID, image):
        self.radiographID = radiographID
        self.toothID = toothID
        self.image = image  # type: image


def loadAllForRadiograph(radiographID):
    """
    Loads all the segments for a given radiograph.
    :return: Dictionary of toothNumber -> Segment
    """
    segments = {}

    for filepath in getSegmentationFilenames(radiographID):
        filename = os.path.split(filepath)[-1]
        toothNumber = int(re.match("{}-([0-9]).png".format(radiographID), filename).group(1))
        segments[toothNumber] = Segment(radiographID, toothNumber, Image.open(filepath))

    return segments
