import numpy as np
import scipy.spatial.distance
from scipy import linalg

import procrustes_analysis
from Landmark import Landmark


class Model:
    def __init__(self, name, landmarks, pcaComponents=20):
        self.name = name
        self.landmarks = landmarks
        self.meanLandmark = None  # type: Landmark
        self.preprocessedLandmarks = []
        self.pcaComponents = pcaComponents
        self.eigenvalues = np.array([])
        self.eigenvectors = np.array([])
        self.meanTheta = None
        self.meanScale = None
        self.sampleAmount = 10
        self.grayLevelModels = {}
        self.normalizedGrayLevelModels = {}
        self.grayLevelModelsInverseCovariances = {}

    def doProcrustesAnalysis(self):
        # procrustes_analysis.drawLandmarks(self.landmarks, "before")

        self.preprocessedLandmarks, self.meanLandmark, self.meanScale, self.meanTheta \
            = procrustes_analysis.performProcrustesAnalysis(self.landmarks)

        # procrustes_analysis.drawLandmarks(self.preprocessedLandmarks, "after")
        return self

    def getTranslatedMean(self, x, y):
        """ Returns the mean landmark translated to x and y. """
        return self.meanLandmark.translate(x, y)

    def getTranslatedAndInverseScaledMean(self, x, y):
        """ Returns the mean landmark rescaled back from unit variance (after procrustes) and translated to x and y. """
        return self.meanLandmark.scale(self.meanScale).translate(x, y)

    def doPCA(self):
        """ Perform PCA on the landmarks after procrustes analysis and store the eigenvalues and eigenvectors. """
        data = [l.getPointsAsList() for l in self.preprocessedLandmarks]
        data.append(data[0])

        S = np.cov(np.transpose(data))

        eigenvalues, eigenvectors = np.linalg.eig(S)
        sorted_values = np.flip(eigenvalues.argsort(), 0)[:self.pcaComponents]

        self.eigenvalues = eigenvalues[sorted_values]
        self.eigenvectors = eigenvectors[:, sorted_values]
        return self

    def buildGrayLevelModels(self):
        """
        Builds gray level models for each of the tooth's landmark points.
        Build gray level models for each of the mean landmark points by averaging the gray level profiles for each
        point of each landmark.
        """
        self.grayLevelModels = {}
        self.normalizedGrayLevelModels = {}
        self.grayLevelModelsInverseCovariances = {}

        for i in range(len(self.meanLandmark.getPointsAsTuples())):
            self.normalizedGrayLevelModels[i] = []

            # Build gray level model for each landmark and add it
            for landmark in self.landmarks:
                grayLevelProfiles, normalizedGrayLevelProfiles, _ = \
                    landmark.grayLevelProfileForAllPoints(self.sampleAmount)

                for pointIndex, profile in grayLevelProfiles.items():
                    if pointIndex not in self.grayLevelModels:
                        self.grayLevelModels[pointIndex] = np.zeros(profile.shape)
                    self.grayLevelModels[pointIndex] += profile

                    self.normalizedGrayLevelModels[i].append(normalizedGrayLevelProfiles[pointIndex])

        for pointIndex, summ in self.grayLevelModels.items():
            self.grayLevelModels[pointIndex] = summ / len(self.meanLandmark.getPointsAsTuples())

            self.grayLevelModelsInverseCovariances[pointIndex] = linalg.inv(
                np.cov(np.transpose(self.normalizedGrayLevelModels[pointIndex]))
            )

        return self

    def mahalanobisDistance(self, profile, landmarkPointIndex):
        """
        Returns the squared Mahalanobis distance of a new gray level profile from the built gray level model with index
        landmarkPointIndex.
        """
        Sp = self.grayLevelModelsInverseCovariances[landmarkPointIndex]
        pMinusMeanTrans = (profile - self.grayLevelModels[landmarkPointIndex])

        return pMinusMeanTrans.T @ Sp @ pMinusMeanTrans

    def findBetterFittingLandmark(self, landmark, radiograph):
        """
        Returns a landmark that is a better fit on the image than the given according to the gray level profiles of
        points of the landmark and the mahalanobis distance.
        """
        landmark.radiograph = radiograph

        # Get the gray level profiles of points on normal lines of the landmark's points
        grayLevelProfiles = landmark.getGrayLevelProfilesForAllNormalPoints(self.sampleAmount)

        bestPoints = []
        for pointIdx, profiles in grayLevelProfiles.items():
            distances = []

            for profile, normalPoint in profiles:
                d = self.mahalanobisDistance(profile, pointIdx)
                distances.append((d, normalPoint))
                print("Mahalanobis dist: {}, p: {}".format(d, normalPoint))

            bestPoints.append(min(distances, key=lambda x: x[0])[1])

        return landmark.copy(np.asarray(bestPoints).flatten())

    def reconstructLandmarkForCoefficients(self, b):
        return Landmark(self.meanLandmark.points + (self.eigenvectors @ b).flatten())

    def matchModelPointsToTargetPoints(self, b, landmarkY):
        # procrustes_analysis.drawLandmarks([landmarkY], "landmarkY")

        # 1. initialize the shape parameters, b, to zero
        # b = np.zeros((self.pcaComponents, 1))
        # diff = float("inf")
        # i = 0

        # while diff > 1:
        # i += 1

        # Generate model points using x = x' + Pb
        x = self.reconstructLandmarkForCoefficients(b)
        # procrustes_analysis.drawLandmarks([x], "x")

        # Project Y into the model coordinate frame by superimposition and return the parameters of the transformation
        y, (translateX, translateY), scale, theta = landmarkY.superimpose(x)
        print((translateX, translateY), scale, theta)
        # procrustes_analysis.drawLandmarks([y], "y")

        # TODO ??? Project y into the tangent plane to x_mean by scaling: y' = y / (y x_mean)
        y.points = y.points / np.dot(y.points, self.meanLandmark.points)

        # Update the model parameters b
        print((y.points - self.meanLandmark.points))
        newB = self.eigenvectors.T @ (y.points - self.meanLandmark.points)
        newB = newB.reshape((self.pcaComponents, -1))
        for i in range(len(newB)):
            limit = 2*np.sqrt(self.eigenvalues[i])
            prev = newB[i]
            newB[i] = np.clip(newB[i], -limit, limit)
            print("prev: {}, now: {}, {}".format(prev, newB[i], limit))


        diff = scipy.spatial.distance.euclidean(b, newB)
        # b = newB
        print("New model points diff:", diff)

        # procrustes_analysis.drawLandmarks([newLandmark], "newLandmark")
        newLandmark = self.reconstructLandmarkForCoefficients(newB).rotate(-theta).scale(scale).translate(-translateX, -translateY)

        return newB, newLandmark

    def reconstruct(self):
        """
        Reconstructs a landmark.
        Be sure to create b for a preprocessed landmark. PCA is done on preprocessed landmarks.
        """
        landmark = self.preprocessedLandmarks[0]
        b = self.eigenvectors.T @ (landmark.points - self.meanLandmark.points)
        b = b.reshape((self.pcaComponents, -1))

        reconstructed = self.reconstructLandmarkForCoefficients(b)

        procrustes_analysis.drawLandmarks([landmark], "origin")
        procrustes_analysis.drawLandmarks([reconstructed], "reconstructed")
        return reconstructed
