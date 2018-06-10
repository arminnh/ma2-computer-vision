import os
import re

from PIL import Image

import util


class Segment:
    def __init__(self, radiographFilename, toothNumber, image):
        self.radiographFilename = radiographFilename
        self.toothNumber = toothNumber
        self.image = image  # type: image


def loadAllForRadiograph(radiographFilename):
    """
    Loads all the segments for a given radiograph.
    :return: Dictionary of toothNumber -> Segment
    """
    segments = {}

    for filepath in util.getSegmentationFilenames(radiographFilename):
        filename = os.path.split(filepath)[-1]
        toothNumber = int(re.match("[0-9]{2}-([0-7]).png", filename).group(1)) + 1
        segments[toothNumber] = Segment(radiographFilename, toothNumber, Image.open(filepath))

    return segments
