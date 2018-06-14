import math

import numpy as np
import scipy.spatial.distance
from scipy import linalg

import procrustes_analysis
from InitializationModel import InitializationModel
from Landmark import Landmark


class ToothModel:
    def __init__(self, name, landmarks, pcaComponents, sampleAmount, projectY=False):
        self.name = name
        self.landmarks = landmarks
        self.preprocessedLandmarks = []
        self.meanLandmark = None  # type: Landmark
        self.meanTheta = None
        self.meanScale = None
        self.pcaComponents = pcaComponents
        self.eigenvalues = np.array([])
        self.eigenvectors = np.array([])
        self.sampleAmount = sampleAmount
        self.y_ij = {}
        self.y_j_bar = {}
        self.C_yj = {}
        self.initializationModel = InitializationModel(landmarks, 28)  # TODO
        self.projectYIntoTangentPlane = projectY

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
        # print(self.eigenvalues)
        return self

    def buildGrayLevelModels(self):
        """
        Builds gray level models for each of the tooth's landmark points.
        Build gray level models for each of the mean landmark points by averaging the gray level profiles for each
        point of each landmark.
        """
        self.C_yj = {}
        self.y_j_bar = {}
        # Build gray level model for each landmark and add it
        for i, landmark in enumerate(self.landmarks):
            grayLevelProfiles, normalizedGrayLevelProfiles, _ = \
                landmark.grayLevelProfileForAllPoints(self.sampleAmount)
            for j, profile in grayLevelProfiles.items():
                if j not in self.y_j_bar:
                    self.y_j_bar[j] = []

                self.y_j_bar[j].append(normalizedGrayLevelProfiles[j])
                # self.y_ij[i][j] = normalizedGrayLevelProfiles[j]

        # for i in range(len(self.meanLandmark.getPointsAsTuples())):
        #     self.normalizedGrayLevelModels[i] = []
        #
        #     # Build gray level model for each landmark and add it
        #     for landmark in self.landmarks:
        #         grayLevelProfiles, normalizedGrayLevelProfiles, _ = \
        #             landmark.grayLevelProfileForAllPoints(self.sampleAmount)
        #
        #         for pointIndex, profile in grayLevelProfiles.items():
        #             if pointIndex not in self.grayLevelModels:
        #                 self.grayLevelModels[pointIndex] = np.zeros(profile.shape)
        #             self.grayLevelModels[pointIndex] += profile
        #
        #             self.normalizedGrayLevelModels[i].append(normalizedGrayLevelProfiles[pointIndex])

        for j in range(len(self.y_j_bar)):
            cov = np.cov(np.transpose(self.y_j_bar[j]))
            self.y_j_bar[j] = np.mean(self.y_j_bar[j], axis=0)
            # cov = np.zeros((self.sampleAmount-1, self.sampleAmount-1))
            # for i in range(len(self.landmarks)):
            #     p = (self.y_ij[i][j] - self.y_j_bar[j])
            #     p.resize(len(p), 1)
            #     res = np.matmul(p, p.T)
            #     cov += res
            # cov /= len(self.landmarks)
            # cov = np.cov(self.y_ij)
            self.C_yj[j] = linalg.pinv(cov)

        return self

    def mahalanobisDistance(self, profile, landmarkPointIndex):
        """
        Returns the squared Mahalanobis distance of a new gray level profile from the built gray level model with index
        landmarkPointIndex.
        """
        Sp = self.C_yj[landmarkPointIndex]
        pMinusMeanTrans = (profile - self.y_j_bar[landmarkPointIndex])

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
        for pointIdx in range(len(grayLevelProfiles)):

            profiles = grayLevelProfiles[pointIdx]
            distances = []

            for profile, normalPoint, _ in profiles:
                d = self.mahalanobisDistance(profile, pointIdx)
                distances.append((abs(d), normalPoint))
                # print("Mahalanobis dist: {:.2f}, p: {}".format(abs(d), normalPoint))

            bestPoints.append(min(distances, key=lambda x: x[0])[1])

        return landmark.copy(np.asarray(bestPoints).flatten())

    def reconstructLandmarkForCoefficients(self, b):
        return Landmark(self.meanLandmark.points + (self.eigenvectors @ b).flatten())

    def matchModelPointsToTargetPoints(self, landmarkY):
        """
        A simple iterative approach towards finding the best pose and shape parameters to match a model instance X to a
        new set of image points Y.
        """
        b = np.zeros((self.pcaComponents, 1))
        diff = float("inf")
        translateX = 0
        translateY = 0
        theta = 0
        scale = 0

        while diff > 1e-9:
            # Generate model points using x = x' + Pb
            x = self.reconstructLandmarkForCoefficients(b)

            # Project Y into the model coordinate frame by superimposition
            # and get the parameters of the transformation
            y, (translateX, translateY), scale, theta = landmarkY.superimpose(x)

            # Project y into the tangent plane to x_mean by scaling: y' = y / (y x_mean)
            # As in "An Introduction to Active Shape Models" by Tim Cootes
            if self.projectYIntoTangentPlane:
                y.points /= y.points.dot(self.meanLandmark.points)

            # Update the model parameters b
            newB = (self.eigenvectors.T @ (y.points - self.meanLandmark.points)).reshape((self.pcaComponents, -1))

            # Constrain the coefficients to lie within certain limits
            for i in range(len(newB)):
                limit = 2 * np.sqrt(abs(self.eigenvalues[i]))
                newB[i] = np.clip(newB[i], -limit, limit)

            diff = scipy.spatial.distance.euclidean(b, newB)
            b = newB

        return self.reconstructLandmarkForCoefficients(b) \
            .rotate(-theta).scale(scale).translate(-translateX, -translateY)

    def reconstruct(self):
        """
        Reconstructs a landmark.
        Be sure to create b for a preprocessed landmark. PCA is done on preprocessed landmarks.
        """
        landmark = self.preprocessedLandmarks[0]
        b = self.eigenvectors.T @ (landmark.points - self.meanLandmark.points)
        b = b.reshape((self.pcaComponents, -1))

        reconstructed = self.reconstructLandmarkForCoefficients(b)

        procrustes_analysis.plotLandmarks([landmark], "origin")
        procrustes_analysis.plotLandmarks([reconstructed], "reconstructed")
        return reconstructed
