import os
from typing import Dict

import Landmark
import Segment
import util


class Radiograph:

    def __init__(self, filename, image, imageUpperJaw, imageLowerJaw, jawSplitLine, landmarks, segments,
                 mirrored=False):
        """
        :param filename: the filename/id of the Radiograph
        """
        self.filename = filename
        self.image = image
        self.imageUpperJaw = imageUpperJaw
        self.imageLowerJaw = imageLowerJaw
        self.jawSplitLine = jawSplitLine
        self.landmarks = landmarks  # type: Dict[int, Landmark.Landmark]
        self.segments = segments  # type: Dict[int, Segment.Segment]
        self.mirrored = mirrored
        for landmark in self.landmarks.values():
            landmark.radiograph = self

    def getLandmarksForTeeth(self, toothNumbers):
        return [v for k, v in self.landmarks if k in toothNumbers]

    def showRaw(self):
        """ Shows the radiograph """
        self.image.show()

    def showWithLandMarks(self):
        raise Exception("todo")
        # img = self.image.copy()
        # draw = ImageDraw.Draw(img)
        #
        # for toothNumber, landmark in self.landmarks.items():
        #     # PIL can't work with numpy arrays so convert to list of tuples
        #     p = landmark.getPointsAsList()
        #     p = [(float(p[2 * j]), float(p[2 * j + 1])) for j in range(int(len(p) / 2))]
        #     draw.line(p + [p[0]], fill="red", width=2)
        #     for i, point in enumerate(p):
        #         draw.text(point, str(i))

    def plotLandMarksWithGrayLevelModels(self):
        raise Exception("todo")
        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.figure()
        #
        # for toothNumber, landmark in self.landmarks.items():
        #     # PIL can't work with numpy arrays so convert to list of tuples
        #     points = landmark.getPointsAsTuples()
        #     X = points[:, 0]
        #     Y = points[:, 1]
        #     plt.plot(X, Y, 'x', label="tooth " + str(toothNumber))
        #     for i in range(len(points)):
        #         plt.text(X[i] - 10, Y[i], i)
        #
        #     normals = landmark.normalSamplesForAllPoints(util.SAMPLE_AMOUNT)
        #     grayLevelProfiles, normalizedGrayLevelProfiles, normalPointsOfLandmarkNr = landmark.grayLevelProfileForAllPoints(
        #         util.SAMPLE_AMOUNT)
        #
        #     for i, profile in normalizedGrayLevelProfiles.items():
        #         # plt.plot(normals[i][0], normals[i][1])
        #
        #         Xs = np.arange(X[i] - util.SAMPLE_AMOUNT, X[i] + util.SAMPLE_AMOUNT)
        #         y = np.repeat(Y[i], 2 * util.SAMPLE_AMOUNT)
        #         profile = profile + abs(profile.min())
        #         profile = [str(p) for p in profile]
        #         plt.scatter(x=Xs, y=y, c=profile, s=10, zorder=3)
        #
        # plt.legend()
        # plt.title("Tooters of radiograph {}".format(self.filename))
        # ax = plt.gca()
        # ax.set_ylim(ax.get_ylim()[::-1])
        # plt.axis("equal")
        # plt.show()

    def showWithSegments(self):
        for segment in self.segments.values():
            segment.image.show()

    def save_img(self):
        self.image.save(
            "/Users/thierryderuyttere/Downloads/pycococreator-master/examples/shapes/train/" + "{}.jpg".format(
                self.filename))


def getRadiographs(numbers=None, extra=False):
    numbers = ["%02d" % n for n in numbers] if numbers is not None else [numbers]
    radiographs = []

    for n in numbers:
        for filepath in util.getRadiographFilenames(n, extra):
            filename = os.path.splitext(os.path.split(filepath)[-1])[0]
            print("Loading radiograph {}, {}".format(n, filepath))

            # Load the radiograph in as is
            img, imgUpperJaw, imgLowerJaw, jawSplitLine, XOffset, YOffset = util.loadRadiographImage(filename)
            segments = Segment.loadAllForRadiograph(filename)
            radiographs.append(Radiograph(
                filename=filename,
                image=img,
                imageUpperJaw=imgUpperJaw,
                imageLowerJaw=imgLowerJaw,
                jawSplitLine=jawSplitLine,
                landmarks=Landmark.loadAllForRadiograph(filename, XOffset, YOffset),
                segments=segments
            ))

            # TODO: do image mirroring in radiograph
            # Load a mirrored version
            # mirrorXOffset = img.size[0]
            # landmarks = {}
            # for k, v in Landmark.loadAllForRadiograph(int(filename) + 14).items():
            #     # Correct mirrored landmark toothNumber and add offset to X values to put landmark in correct position
            #     landmarks[util.flipToothNumber(v.toothNumber)] = v.addToXValues(mirrorXOffset)
            #
            # for segment in segments.values():
            #     segment.image = ImageOps.mirror(segment.image)
            #     segment.toothNumber = util.flipToothNumber(segment.toothNumber)
            #
            # radiographs.append(Radiograph(
            #     filename=filename,
            #     image=ImageOps.mirror(img),
            #     landmarks=landmarks,
            #     segments=segments,
            #     mirrored=True
            # ))

    return radiographs


def getAllLandmarksInRadiographs(radiographs):
    landmarks = []
    for r in radiographs:
        landmarks += list(r.landmarks.values())
    return landmarks
