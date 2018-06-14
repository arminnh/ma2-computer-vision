import numpy as np
from scipy import linalg

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
                                                                    verticalPoints, landmark.radiograph)
            normalized.append(pixelProfile)
            self.profileForImage[i] = list(zip(b1, horizontalPoints)) + list(zip(b2, verticalPoints)) + list(
                zip(b3, diagonalPoints1)) + list(zip(b4, diagonalPoints2))

        self.y_bar = np.mean(normalized, 0)
        self.covInv = linalg.pinv(np.cov(np.transpose(normalized)))

    def getInitPixelProfile(self, diagonalPoints1, diagonalPoints2, horizontalPoints, verticalPoints, radiograph):
        b1, n1, s1 = util.getPixels(radiograph, horizontalPoints, True)
        b2, n2, s2 = util.getPixels(radiograph, verticalPoints, True)
        b3, n3, s3 = util.getPixels(radiograph, diagonalPoints1, True)
        b4, n4, s4 = util.getPixels(radiograph, diagonalPoints2, True)
        pixelProfile = list(s1) + list(s2) + list(s3) + list(s4)
        return b1, b2, b3, b4, pixelProfile

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

    def getBetterOrigin(self, currentOriginForModel, radiograph):
        diagonalPoints1, diagonalPoints2, horizontalPoints, verticalPoints = self.sampleInitProfilePoints(
            currentOriginForModel)

        points = diagonalPoints1 + diagonalPoints2 + horizontalPoints + verticalPoints
        distances = []
        for p in points:
            d1, d2, h1, v1 = self.sampleInitProfilePoints(p)
            _, _, _, _, pixelProfile = self.getInitPixelProfile(d1, d2, h1, v1, radiograph)
            distances.append((self.mahalanobisDistance(np.asarray(pixelProfile)), p))

        return min(distances, key=lambda x: x[0])[1]
