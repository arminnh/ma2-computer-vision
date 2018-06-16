import cv2
import numpy as np

import util
from models import TeethActiveShapeModel


class MultiResolutionGUI:

    def __init__(self, radiographs, teethActiveShapeModel):
        self.name = "Computer Vision KU Leuven"
        self.radiographs = radiographs
        self.currentRadiograph = None
        self.currentRadiographIndex = 0
        self.model = teethActiveShapeModel  # type: TeethActiveShapeModel
        self.currentResolutionLevel = self.model.resolutionLevels - 1
        self.img = None
        self.currentLandmark = None

    def open(self):
        self.createWindow()
        self.setCurrentImage(self.currentRadiographIndex)
        self.initTrackBars()

        while True:
            # clone the img
            img = self.img.copy()

            y, x = img.shape
            cv2.line(img, (int(x / 2), 0), (int(x / 2), int(y)), 255, 1)
            cv2.line(img, (0, int(y / 2)), (x, int(y / 2)), 255, 1)

            if self.currentRadiographIndex < len(self.model.mouthLandmarks):
                landmark = self.model.mouthLandmarks[self.currentRadiographIndex]
                landmark = landmark.scale(0.5 ** self.currentResolutionLevel)
                self.drawLandmark(landmark, color=130)

            cv2.imshow(self.name, img)

            # Key Listeners
            pressed_key = cv2.waitKey(50)

            if pressed_key == ord("x") or pressed_key == ord("c"):
                if len(self.radiographs) > 1:
                    if pressed_key == ord("x"):
                        self.increaseRadiographIndex(-1)
                    elif pressed_key == ord("c"):
                        self.increaseRadiographIndex(1)
                    cv2.setTrackbarPos("radiograph", self.name, self.currentRadiographIndex)

            elif pressed_key == ord("n"):
                if self.currentLandmark is None:
                    cv2.displayOverlay(self.name, "No landmark set yet!", 1000)
                else:
                    self.findBetterLandmark()

            elif pressed_key == ord("u"):
                cv2.setTrackbarPos("resolution level", self.name, self.currentResolutionLevel + 1)

            elif pressed_key == ord("d"):
                cv2.setTrackbarPos("resolution level", self.name, self.currentResolutionLevel - 1)

            if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

    def createWindow(self):
        cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.name, 1000, 700)
        cv2.setMouseCallback(self.name, self.mouseListener)
        return self

    # mouse callback function
    def mouseListener(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.currentLandmark = self.model.getTranslatedAndInverseScaledMeanMouth(self.currentResolutionLevel, x, y)

            self.drawLandMarkWithNormals(self.currentLandmark, withNormalLines=True, withGrayLevels=True)

    def refreshOverlay(self):
        cv2.displayOverlay(self.name, "Showing image {}".format(self.currentRadiographIndex), 1000)
        return self

    def refreshCurrentImage(self):
        self.img = self.currentRadiograph.imgPyramid[self.currentResolutionLevel].copy()

        jawSplitLine = self.currentRadiograph.jawSplitLine
        for i, (x, y) in enumerate(jawSplitLine):
            if i > 0:
                y = int(round(y * 0.5 ** self.currentResolutionLevel))
                prevY = int(round(jawSplitLine[i - 1][1] * 0.5 ** self.currentResolutionLevel))

                cv2.line(self.img, (jawSplitLine[i - 1][0], prevY), (x, y), 255, 1)

    def setCurrentImage(self, idx):
        self.currentRadiographIndex = idx
        self.currentRadiograph = self.radiographs[idx]
        self.refreshCurrentImage()
        self.refreshOverlay()
        return self

    def setCurrentResolutionLevel(self, level):
        self.currentResolutionLevel = level

        if self.currentResolutionLevel < 0:
            self.currentResolutionLevel = 0

        if self.currentResolutionLevel >= self.model.resolutionLevels:
            self.currentResolutionLevel = self.model.resolutionLevels - 1

        self.refreshCurrentImage()
        self.refreshOverlay()
        return self

    def initTrackBars(self):
        if len(self.radiographs) > 1:
            cv2.createTrackbar("radiograph", self.name, self.currentRadiographIndex, len(self.radiographs) - 1,
                               self.setCurrentImage)

        cv2.createTrackbar("resolution level", self.name, self.currentResolutionLevel, self.model.resolutionLevels - 1,
                           self.setCurrentResolutionLevel)

        return self

    def increaseRadiographIndex(self, amount):
        self.currentRadiographIndex += amount

        if self.currentRadiographIndex < 0:
            self.currentRadiographIndex = 0

        if self.currentRadiographIndex >= len(self.radiographs):
            self.currentRadiographIndex = len(self.radiographs) - 1

        return self

    def drawLandmark(self, landmark, color):
        points = landmark.getPointsAsTuples().round().astype(int)

        for i in range(int(len(points) / 40)):
            subpoints = points[i * 40:(i + 1) * 40]

            for j, (x, y) in enumerate(subpoints):
                (x2, y2) = subpoints[(j + 1) % len(subpoints)]

                cv2.line(self.img, (x, y), (x2, y2), color, 1)

    def drawLandMarkWithNormals(self, landmark, color=255, withNormalLines=False, withGrayLevels=False):
        points = landmark.getPointsAsTuples().round().astype(int)

        # draw normal lines
        if withNormalLines:
            for i in range(int(len(points) / 40)):
                subpoints = points[i * 40:(i + 1) * 40]

                for j in range(len(subpoints)):
                    m = util.getNormalSlope(subpoints[j - 1], subpoints[j], subpoints[(j + 1) % len(subpoints)])
                    normalPoints = np.asarray(util.sampleLine(m, subpoints[j], self.model.sampleAmount))

                    for x, y in normalPoints:
                        cv2.line(self.img, (x, y), (x, y), 255, 1)

        # draw gray level profiles
        if withGrayLevels:
            profilesForLandmarkPoints = landmark.getGrayLevelProfilesForNormalPoints(
                img=self.currentRadiograph.imgPyramid[self.currentResolutionLevel].copy(),
                sampleAmount=self.model.sampleAmount,
                grayLevelModelSize=self.model.grayLevelModelSize,
                derive=False
            )

            for landmarkPointIdx in range(len(profilesForLandmarkPoints)):
                for profileContainer in profilesForLandmarkPoints[landmarkPointIdx]:
                    grayLevelProfile = profileContainer["grayLevelProfile"]
                    grayLevelProfilePoints = profileContainer["grayLevelProfilePoints"]
                    self.drawGrayLevelProfile(grayLevelProfilePoints, grayLevelProfile)

        self.drawLandmark(landmark, color)

    def findBetterLandmark(self):
        """
        Execute an iteration of "matching model points to target points"
        model points are defined by the model, target points by the 'bestLandmark'
        """
        previousLandmark = self.currentLandmark
        d = 2
        i = 0

        while d > 1 and i < 1:
            newTargetLandmark = self.model.findBetterFittingLandmark(
                resolutionLevel=self.currentResolutionLevel,
                img=self.currentRadiograph.imgPyramid[self.currentResolutionLevel].copy(),
                landmark=previousLandmark
            )

            improvedLandmark = self.model.matchModelPointsToTargetPoints(newTargetLandmark)

            d = improvedLandmark.shapeDistance(previousLandmark)
            previousLandmark = improvedLandmark

            i += 1
            print("Improvement iteration {}, distance {}".format(i, d))

        self.currentLandmark = previousLandmark
        self.refreshCurrentImage()
        self.drawLandmark(self.currentLandmark, 255)

    def drawGrayLevelProfile(self, points, profile):
        for i, point in enumerate(points):
            cv2.line(self.img, point, point, int(profile[i]), 1)
