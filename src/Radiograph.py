import os
from typing import Dict

from PIL import ImageDraw, Image

import Landmark
import Segment
import util


class Radiograph:

    def __init__(self, filename):
        """
        :param filename: the filename/id of the Radiograph
        """
        self.filename = filename
        self.image = util.loadRadiographImage(filename)  # type: Image
        self.landMarks = Landmark.loadAllForRadiograph(filename)  # type: Dict[int, Landmark.Landmark]
        self.segments = Segment.loadAllForRadiograph(filename)  # type: Dict[int, Segment.Segment]

    def getLandmarksForTeeth(self, toothNumbers):
        return [v for k, v in self.landMarks if k in toothNumbers]

    def showRawRadiograph(self):
        """ Shows the radiograph """
        self.image.show()

    def showRadiographWithLandMarks(self):
        img = self.image.copy()
        draw = ImageDraw.Draw(img)

        for k, l in self.landMarks.items():
            p = l.getPointsAsTuples().flatten()
            # Apparently PIL can't work with numpy arrays...
            p = [(float(p[2 * j]), float(p[2 * j + 1])) for j in range(int(len(p) / 2))]
            draw.line(p + [p[0]], fill="red", width=2)

        img.show()

    def showSegmentationForTooth(self, toothNumber):
        if toothNumber in self.segments:
            self.segments[toothNumber].image.show()

    def preprocessRadiograph(self, transformations):
        for transform in transformations:
            self.image = transform(self.image)

    def save_img(self):
        self.image.save(
            "/Users/thierryderuyttere/Downloads/pycococreator-master/examples/shapes/train/" + "{}.jpg".format(
                self.filename))


def getAllRadiographs():
    radiographs = []

    for filepath in util.getRadiographFilenames():
        filename = os.path.splitext(os.path.split(filepath)[-1])[0]

        radiographs.append(Radiograph(filename))

    return radiographs


def getAllLandmarksInRadiographs(radiographs):
    landmarks = []
    for r in radiographs:
        landmarks += list(r.landMarks.values())
    return landmarks
