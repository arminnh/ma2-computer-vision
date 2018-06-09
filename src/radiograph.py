from typing import Dict
import numpy as np
from PIL import Image, ImageDraw
import os, inspect
import glob

from src.landmark import Landmark
import re


UPPER_TEETH = {1, 2, 3, 4}
LOWER_TEETH = {5, 6, 7, 8}
LEFT_TEETH = {1, 2, 5, 6}
RIGHT_TEETH = {3, 4, 7, 8}

here = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

class Radiograph:

    def __init__(self, radioID):
        """
        Initialises a radiograph object
        :param radioID: the id of the radiograph
        """

        # "%02d" % radioID => creates a double digit numberstring, ex: radioID = 1 then this value is "01"
        self.radioID = "%02d" % radioID
        self.photo = self._load_tif(self.radioID)
        self.landMarks = self._load_landmarks(radioID)  # type: Dict[int, Landmark]
        self.segmentations = self._load_segmentations(self.radioID)

    def getTeeth(self, toothNumbers):
        return [v for k, v in self.landMarks if k in toothNumbers]

    def _load_segmentations(self, radioID):
        segmLoc = here + "/../resources/data/segmentations/"
        segmName = "{}-*.png".format(radioID)

        allSegms = glob.glob(segmLoc + segmName)
        segmentations = {}
        for segm in allSegms:
            fileName = segm.split("/")[-1]
            nr = int(re.match("{}-([0-9]).png".format(radioID), fileName).group(1))
            segmentations[nr] = Image.open(segm)

        return segmentations

    def _load_tif(self, radioID):
        """
        Loads a tif file from the radiographs folder
        :param radioID: the id of the radiograph
        :return: returns the radiograph image
        """
        radiographs_ = "/../resources/data/radiographs"

        # Get all tifs in radiographs (except extra's)
        # These are the full path to these files...
        tifs = glob.glob(here + radiographs_ + "/*.tif")

        # So remove the full path and just keep the file
        tif_names = [x.split("/")[-1] for x in tifs]

        # Check if the tif of the current radioID is present in our current tifs
        if "{}.tif".format(radioID) in tif_names:
            return Image.open(here + "{}/{}.tif".format(radiographs_, radioID))
        else:
            # Okay so maybe this is an extra?
            return Image.open(here + "{}/extra/{}.tif".format(radiographs_, radioID))

    def _load_landmarks(self, radioID):
        """
        Loads all the landmarks on a radiograph for a given radioID
        :param radioID: the id of the radiograph
        :return: returns a dictionary of ID -> landmark
        """
        landmarkLoc = here + "/../resources/data/landmarks"
        landmarkName = "landmarks{}-*.txt".format(radioID)

        allLandMarks = glob.glob(landmarkLoc + "/original/" + landmarkName) + glob.glob(landmarkLoc + "/mirrored/" + landmarkName)

        landMarks = {}
        for landMark in allLandMarks:
            fileName = landMark.split("/")[-1]
            nr = int(re.match("landmarks{}-([0-9]).txt".format(radioID), fileName).group(1))
            landMarks[nr] = Landmark(nr, filename=landMark)

        return landMarks

    def showRawRadiograph(self):
        """
        Shows the radiograph
        :return:
        """
        self.photo.show()

    def showRadiographWithLandMarks(self):
        img = self.photo.copy()
        draw = ImageDraw.Draw(img)
        for k, l in self.landMarks.items():
            p = l.getPointsAsTuples().flatten()
            # Apparently PIL can't work with numpy arrays...
            p = [(float(p[2*j]),float(p[2*j+1])) for j in range(int(len(p)/2))]
            draw.line(p + [p[0]], fill="red", width=2)

        img.show()

    def showSegmentationNr(self, nr):
        if nr in self.segmentations:
            self.segmentations[nr].show()


    def preprocessRadiograph(self, transformations):
        for transform in transformations:
            self.photo = transform(self.photo)

    def save_img(self):
        self.photo.save("/Users/thierryderuyttere/Downloads/pycococreator-master/examples/shapes/train/" + "{}.jpg".format(self.radioID))