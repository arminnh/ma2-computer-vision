import os
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np

import Landmark
import Segment
import images
import util


class Radiograph:

    def __init__(self, filename, imgPyramid, imgUpperJaw, imgLowerJaw, jawSplitLine, landmarks, segments, mirrored=False):
        """
        :param filename: the filename/id of the Radiograph
        """
        self.filename = filename
        self.imgPyramid = imgPyramid
        self.imgUpperJaw = imgUpperJaw
        self.imgLowerJaw = imgLowerJaw
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
        windowName = "Radiograph {}".format(self.filename)
        cv2.imshow(windowName, self.imgPyramid[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def showWithLandMarks(self):
        img = self.imgPyramid[0].copy()

        for toothNumber, landmark in self.landmarks.items():
            points = landmark.getPointsAsTuples()

            for i, (x, y) in enumerate(points):
                (x2, y2) = points[(i + 1) % len(points)]
                cv2.line(img, (int(x), int(y)), (int(x2), int(y2)), 255, 3)

                cv2.putText(img, str(i), (int(x), int(y)), 2, 0.4, 180)

        windowName = "Radiograph {} with landmarks".format(self.filename)
        cv2.imshow(windowName, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plotLandMarksWithGrayLevelModels(self):
        plt.figure()

        for toothNumber, landmark in self.landmarks.items():
            points = landmark.getPointsAsTuples()
            X = points[:, 0]
            Y = points[:, 1]

            plt.plot(X, Y, 'x', label="tooth " + str(toothNumber))
            for i in range(len(points)):
                plt.text(X[i] - 10, Y[i], i)

            normalizedGrayLevelProfiles = landmark.normalizedGrayLevelProfilesForLandmarkPoints(
                img=self.imgPyramid[0],
                grayLevelModelSize=util.SAMPLE_AMOUNT,
            )

            for i, profile in normalizedGrayLevelProfiles.items():
                # plt.plot(normals[i][0], normals[i][1])

                m = util.getNormalSlope(points[i - 1], points[i], points[(i + 1) % len(points)])
                normalPoints = np.asarray(util.sampleLine(m, points[i], pixelsToSample=util.SAMPLE_AMOUNT))
                normalX = normalPoints[:, 0]
                normalY = normalPoints[:, 1]

                profile = profile + abs(profile.min())
                profile = [str(p) for p in profile]
                plt.scatter(x=normalX, y=normalY, c=profile, s=2, zorder=3)

        plt.legend()
        plt.title("Teeth of radiograph {}".format(self.filename))
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.axis("equal")
        plt.show()

    def showWithSegments(self):
        raise Exception("todo")
        # for segment in self.segments.values():
        #     segment.imgshow()


def getRadiographs(numbers=None, extra=False, resolutionLevels=4):
    numbers = ["%02d" % n for n in numbers] if numbers is not None else []
    radiographs = []

    for n in numbers:
        for filepath in util.getRadiographFilenames(n, extra):
            filename = os.path.splitext(os.path.split(filepath)[-1])[0]
            print("Loading radiograph {}, {}".format(n, filepath))

            # Load the radiograph in as is
            imgPyramid, imgUpperJaw, imgLowerJaw, jawSplitLine, XOffset, YOffset \
                = images.loadRadiographImage(filename, resolutionLevels)
            landmarks = Landmark.loadAllForRadiograph(filename, XOffset, YOffset)
            segments = Segment.loadAllForRadiograph(filename)

            radiographs.append(Radiograph(
                filename=filename,
                imgPyramid=imgPyramid,
                imgUpperJaw=imgUpperJaw,
                imgLowerJaw=imgLowerJaw,
                jawSplitLine=jawSplitLine,
                landmarks=landmarks,
                segments=segments
            ))

            # Load a mirrored version
            xSize = imgPyramid[0].shape[1]

            mirroredLandmarks = {}
            for k, v in landmarks.items():
                mirroredLandmark = v.copy()
                points = mirroredLandmark.getPointsAsTuples()
                for i, (x, y) in enumerate(points):
                    points[i] = (xSize - x, y)
                mirroredLandmark.points = points[::-1].flatten()
                mirroredLandmarks[util.flipToothNumber(v.toothNumber)] = mirroredLandmark

            mirroredImgPyramid = []
            for img in imgPyramid:
                mirroredImgPyramid.append(img[:, ::-1])

            mirroredImgUpperJaw = imgUpperJaw[:, ::-1]
            mirroredImgLowerJaw = imgLowerJaw[:, ::-1]
            mirroredJawSplitLine = jawSplitLine[::-1, :]

            radiographs.append(Radiograph(
                filename=filename,
                imgPyramid=mirroredImgPyramid,
                imgUpperJaw=mirroredImgUpperJaw,
                imgLowerJaw=mirroredImgLowerJaw,
                jawSplitLine=mirroredJawSplitLine,
                landmarks=mirroredLandmarks,
                segments=segments
            ))

    return radiographs
