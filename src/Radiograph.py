import os
from typing import Dict

from PIL import ImageDraw, Image, ImageOps

import Landmark
import Segment
import util


class Radiograph:

    def __init__(self, filename, image, landmarks, segments, mirrored=False):
        """
        :param filename: the filename/id of the Radiograph
        """
        self.filename = filename
        self.image = image  # type: Image
        self.landmarks = landmarks  # type: Dict[int, Landmark.Landmark]
        self.segments = segments  # type: Dict[int, Segment.Segment]
        self.mirrored = mirrored

    def getLandmarksForTeeth(self, toothNumbers):
        return [v for k, v in self.landmarks if k in toothNumbers]

    def showRaw(self):
        """ Shows the radiograph """
        self.image.show()

    def showWithLandMarks(self):
        img = self.image.copy()
        draw = ImageDraw.Draw(img)

        for k, l in self.landmarks.items():
            # PIL can't work with numpy arrays so convert to list of tuples
            p = l.getPointsAsList()
            p = [(float(p[2 * j]), float(p[2 * j + 1])) for j in range(int(len(p) / 2))]
            draw.line(p + [p[0]], fill="red", width=2)

        img.show()

    def showWithSegments(self):
        for segment in self.segments.values():
            segment.image.show()

    def preprocessRadiograph(self, transformations):
        img = self.image
        for transform in transformations:
            img = transform(img)
        return img

    def save_img(self):
        self.image.save(
            "/Users/thierryderuyttere/Downloads/pycococreator-master/examples/shapes/train/" + "{}.jpg".format(
                self.filename))


def getRadiographs(numbers=None, extra=False):
    number = ["%02d" % n for n in numbers] if numbers is not None else []
    radiographs = []

    for n in number:
        for filepath in util.getRadiographFilenames(n, extra):
            filename = os.path.splitext(os.path.split(filepath)[-1])[0]
            print("Loading radiograph {}, {}".format(n, filepath))

            # Load the radiograph in as is
            img = util.loadRadiographImage(filename)
            segments = Segment.loadAllForRadiograph(filename)
            radiographs.append(Radiograph(
                filename=filename,
                image=img,
                landmarks=Landmark.loadAllForRadiograph(filename),
                segments=segments
            ))

            # Load a mirrored version
            mirrorXOffset = img.size[0]
            landmarks = {}
            for k, v in Landmark.loadAllForRadiograph(int(filename) + 14).items():
                # Correct mirrored landmark toothNumber and add offset to X values to put landmark in correct position
                landmarks[util.flipToothNumber(v.toothNumber)] = v.addToXValues(mirrorXOffset)

            for segment in segments.values():
                segment.image = ImageOps.mirror(segment.image)
                segment.toothNumber = util.flipToothNumber(segment.toothNumber)

            radiographs.append(Radiograph(
                filename=filename,
                image=ImageOps.mirror(img),
                landmarks=landmarks,
                segments=segments,
                mirrored=True
            ))

    return radiographs


def getAllLandmarksInRadiographs(radiographs):
    landmarks = []
    for r in radiographs:
        landmarks += list(r.landmarks.values())
    return landmarks
