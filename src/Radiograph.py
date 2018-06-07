from PIL import Image, ImageDraw
import os, inspect
import glob
from Landmark import Landmark
import re

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
        self.landMarks = self._load_landmarks(radioID)
        self.segmentations = self._load_segmentations(self.radioID)

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
            landMarks[nr] = Landmark(landMark, nr)

        return landMarks

    def showRawRadiograph(self):
        """
        Shows the radiograph
        :return:
        """
        self.photo.show()

    def showRadiographWithLandMarks(self):
        draw = ImageDraw.Draw(self.photo)
        for k, landMark in self.landMarks.items():
            points = landMark.getPointsAsTuples()
            draw.line(points + [points[0]], fill=(255,0,0), width=2)

        self.photo.show()

    def showSegmentationNr(self, nr):
        if nr in self.segmentations:
            self.segmentations[nr].show()


    def M(self, s, theta, X):
        pass

