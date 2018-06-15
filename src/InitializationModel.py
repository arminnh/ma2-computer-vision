import numpy as np
from scipy import linalg

import images
import util


class InitializationModel:

    def __init__(self, landmarks, sampleAmount):
        self.landmarks = landmarks
        self.meanOrigin = np.mean([np.mean(l.getPointsAsTuples(), 0) for l in landmarks], 0)
        self.sampleAmount = sampleAmount
        self.y_bar = None
        self.covInv = None
        self.profileForImage = {}
        self.sampleOriginGrayLevels()

    def sampleOriginGrayLevels(self):
        normalized = []
        for i, landmark in enumerate(self.landmarks):
            origin = np.mean(landmark.getPointsAsTuples(), 0)

            diagonalPoints1, diagonalPoints2, horizontalPoints, verticalPoints = self.sampleInitProfilePoints(origin)

            b1, b2, b3, b4, pixelProfile = self.getInitPixelProfile(diagonalPoints1, diagonalPoints2, horizontalPoints,
                                                                    verticalPoints, landmark.getCorrectRadiographPart())
            normalized.append(pixelProfile)
            self.profileForImage[i] = list(zip(b1, horizontalPoints)) + list(zip(b2, verticalPoints)) + list(
                zip(b3, diagonalPoints1)) + list(zip(b4, diagonalPoints2))

        self.y_bar = np.mean(normalized, 0)
        self.covInv = linalg.pinv(np.cov(np.transpose(normalized)))

    def getInitPixelProfile(self, diagonalPoints1, diagonalPoints2, horizontalPoints, verticalPoints, img):
        raw1, normalized1 = images.getPixelProfile(img, horizontalPoints, True)
        raw2, normalized2 = images.getPixelProfile(img, verticalPoints, True)
        raw3, normalized3 = images.getPixelProfile(img, diagonalPoints1, True)
        raw4, normalized4 = images.getPixelProfile(img, diagonalPoints2, True)
        normalizedProfile = list(normalized1) + list(normalized2) + list(normalized3) + list(normalized4)
        return raw1, raw2, raw3, raw4, normalizedProfile

    def sampleInitProfilePoints(self, origin):
        horizontalPoints = util.sampleLine(0, origin, self.sampleAmount)
        verticalPoints = util.sampleLine(1000, origin, self.sampleAmount)
        diagonalPoints1 = util.sampleLine(1, origin, self.sampleAmount)
        diagonalPoints2 = util.sampleLine(-1, origin, self.sampleAmount)
        return diagonalPoints1, diagonalPoints2, horizontalPoints, verticalPoints

    def mahalanobisDistance(self, profile):
        """
        Returns the squared Mahalanobis distance of a new gray level profile from the built gray level model
        """
        pMinusMeanTrans = (profile - self.y_bar)

        return pMinusMeanTrans.T @ self.covInv @ pMinusMeanTrans

    def getBetterOrigin(self, currentOriginForModel, img):
        diagonalPoints1, diagonalPoints2, horizontalPoints, verticalPoints = self.sampleInitProfilePoints(
            currentOriginForModel)

        points = diagonalPoints1 + diagonalPoints2 + horizontalPoints + verticalPoints
        distances = []
        for p in points:
            d1, d2, h1, v1 = self.sampleInitProfilePoints(p)
            _, _, _, _, pixelProfile = self.getInitPixelProfile(d1, d2, h1, v1, img)
            distances.append((self.mahalanobisDistance(np.asarray(pixelProfile)), p))

        return min(distances, key=lambda x: x[0])[1]
