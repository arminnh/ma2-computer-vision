from typing import Dict

from PIL import ImageDraw, Image

import helpers
import landmark
import segment


class Radiograph:

    def __init__(self, radiographID):
        """
        :param radiographID: the id of the Radiograph
        """
        self.id = "%02d" % radiographID  # "%02d" for double digit string, e.g. "01" for radioID 1
        self.image = helpers.loadRadiographImage(self.id)  # type: Image
        self.landMarks = landmark.loadAllForRadiograph(radiographID)  # type: Dict[int, landmark.Landmark]
        self.segmentations = segment.loadAllForRadiograph(radiographID)  # type: Dict[int, segment.Segment]

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
        if toothNumber in self.segmentations:
            self.segmentations[toothNumber].image.show()

    def preprocessRadiograph(self, transformations):
        for transform in transformations:
            self.image = transform(self.image)

    def save_img(self):
        self.image.save(
            "/Users/thierryderuyttere/Downloads/pycococreator-master/examples/shapes/train/" + "{}.jpg".format(self.id))


def getAllRadiographs():
    return [Radiograph(i) for i in range(1, 31)]


def getAllLandmarksInRadiographs(radiographs):
    landmarks = []
    for r in radiographs:
        landmarks += list(r.landMarks.values())
    return landmarks
