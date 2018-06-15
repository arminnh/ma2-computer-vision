import os
import re

import cv2

import util


class Segment:
    def __init__(self, radiographFilename, toothNumber, img):
        self.radiographFilename = radiographFilename
        self.toothNumber = toothNumber
        self.img = img


def loadAllForRadiograph(radiographFilename):
    """
    Loads all the segments for a given radiograph.
    :return: Dictionary of toothNumber -> Segment
    """
    segments = {}

    for filepath in util.getSegmentationFilenames(radiographFilename):
        filename = os.path.split(filepath)[-1]
        toothNumber = int(re.match("[0-9]{2}-([0-7]).png", filename).group(1)) + 1
        img = cv2.imread(filename)
        segments[toothNumber] = Segment(radiographFilename, toothNumber, img)

    return segments
