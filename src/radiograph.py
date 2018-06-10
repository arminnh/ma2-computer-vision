import glob
import inspect
import re
from typing import Dict

import os
from PIL import ImageDraw

import segmentation
from landmark import Landmark
import helpers

here = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

class Radiograph:

    def __init__(self, radiographID):
        """
        :param radiographID: the id of the Radiograph
        """

        # "%02d" % radioID => creates a double digit numberstring, ex: radioID = 1 then this value is "01"
        self.id = "%02d" % radiographID
        self.photo = helpers.getRadiographImage(self.id)
        self.landMarks = self._load_landmarks(radiographID)  # type: Dict[int, Landmark]
        self.segmentations = segmentation.loadForRadiograph(radiographID)

    def getTeeth(self, toothNumbers):
        return [v for k, v in self.landMarks if k in toothNumbers]

    def _load_landmarks(self, radioID):
        """
        Loads all the landmarks on a radiograph for a given radioID
        :param radioID: the id of the radiograph
        :return: returns a dictionary of ID -> landmark
        """
        landmarkLoc = here + "/../resources/data/landmarks"
        landmarkName = "landmarks{}-*.txt".format(radioID)

        allLandMarks = glob.glob(landmarkLoc + "/original/" + landmarkName)
        mirrored = False
        if mirrored:
            allLandMarks += glob.glob(landmarkLoc + "/mirrored/" + landmarkName)

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
            p = [(float(p[2 * j]), float(p[2 * j + 1])) for j in range(int(len(p) / 2))]
            draw.line(p + [p[0]], fill="red", width=2)

        img.show()

    def showSegmentationNr(self, nr):
        if nr in self.segmentations:
            self.segmentations[nr].show()

    def preprocessRadiograph(self, transformations):
        for transform in transformations:
            self.photo = transform(self.photo)

    def save_img(self):
        self.photo.save(
            "/Users/thierryderuyttere/Downloads/pycococreator-master/examples/shapes/train/" + "{}.jpg".format(
                self.id))


def getAllRadiographs():
    return [Radiograph(i) for i in range(1, 31)]
