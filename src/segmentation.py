import re

from PIL import Image

from helpers import getSegmentationFilenames


def loadForRadiograph(radiographID):
    segmentations = {}

    for filename in getSegmentationFilenames(radiographID):
        fileName = filename.split("/")[-1]
        nr = int(re.match("{}-([0-9]).png".format(radiographID), fileName).group(1))
        segmentations[nr] = Image.open(filename)

    return segmentations


class Segmentation:

    def __init__(self, ):
        pass
