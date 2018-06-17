import numpy as np
from scipy import linalg

import images
import util


class CenterInitializationModel:

    def __init__(self, landmarks, sampleAmount):
        self.landmarks = landmarks
        self.sampleAmount = sampleAmount
        self.meanCenter = np.mean([np.mean(l.getPointsAsTuples(), 0) for l in landmarks], 0)
        self.meanGrayLevelProfile = {}
        self.grayLevelProfileCovarianceMatrix = {}
        self.grayLevelProfileForImage = {}

        self.sampleOriginGrayLevels()

    def sampleInitProfilePoints(self, center):
        verticalPoints = util.sampleLine(1000, center, self.sampleAmount * 2)
        diagonalPoints1 = util.sampleLine(0.1, center, self.sampleAmount)
        horizontalPoints = util.sampleLine(0, center, self.sampleAmount)
        diagonalPoints2 = util.sampleLine(-0.1, center, self.sampleAmount)

        return horizontalPoints + verticalPoints + diagonalPoints1 + diagonalPoints2

    def sampleOriginGrayLevels(self):
        normalized = []

        for i, landmark in enumerate(self.landmarks):
            center = np.mean(landmark.getPointsAsTuples(), 0)

            points = self.sampleInitProfilePoints(center)

            rawProfile, normalizedProfile = images.getPixelProfile(landmark.getCorrectRadiographPart(), points,
                                                                   derive=True)
            normalized.append(normalizedProfile)

            self.grayLevelProfileForImage[i] = (points, rawProfile)

        self.meanGrayLevelProfile = np.mean(normalized, 0)
        self.grayLevelProfileCovarianceMatrix = linalg.pinv(np.cov(np.transpose(normalized)))

    def mahalanobisDistance(self, normalizedGrayLevelProfile):
        """
        Returns the squared Mahalanobis distance of a new gray level profile from the built gray level model
        """
        pMinusMeanTrans = (normalizedGrayLevelProfile - self.meanGrayLevelProfile)

        return pMinusMeanTrans.T @ self.grayLevelProfileCovarianceMatrix @ pMinusMeanTrans

    def getBetterCenter(self, img, currentCenter):
        points = self.sampleInitProfilePoints(currentCenter)

        distances = []
        for p in points:
            points2 = self.sampleInitProfilePoints(p)

            _, normalizedProfile = images.getPixelProfile(img, points2, derive=True)

            distances.append((abs(self.mahalanobisDistance(normalizedProfile)), p))

        distance, point = min(distances, key=lambda x: x[0])
        return distance, point
