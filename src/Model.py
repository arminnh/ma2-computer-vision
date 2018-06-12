import numpy as np
import scipy.spatial.distance

import procrustes_analysis
from Landmark import Landmark
from scipy import linalg

class Model:
    def __init__(self, name, landmarks, pcaComponents=20):
        self.name = name
        self.landmarks = landmarks
        self.meanLandmark = None  # type: Landmark
        self.preprocessedLandmarks = []
        self.pcaComponents = pcaComponents
        self.eigenvalues = []
        self.eigenvectors = []
        self.meanTheta = None
        self.meanScale = None
        self.grayLevelModels = {}
        self.sampleAmount = 5

    def doProcrustesAnalysis(self):
        # procrustes_analysis.drawLandmarks(self.landmarks, "before")

        self.preprocessedLandmarks, self.meanLandmark, self.meanTheta, self.meanScale \
            = procrustes_analysis.performProcrustesAnalysis(self.landmarks)

        # procrustes_analysis.drawLandmarks(self.preprocessedLandmarks, "after")
        return self

    def translateMean(self, x, y):
        self.meanLandmark = self.meanLandmark.translatePoints(x, y)
        return self.meanLandmark

    def translateAndRescaleMean(self, x, y):
        self.meanLandmark = self.meanLandmark.normalize()
        self.meanLandmark.points *= self.meanScale
        mean = self.translateMean(x, y)
        return mean

    def doPCA(self):
        data = [l.getPointsAsList() for l in self.preprocessedLandmarks]

        # [m, eigenvalues, self.eigenvectors] = self._pca(data,self.pcaComponents)

        S = np.cov(np.transpose(data))

        eigenvalues, eigenvectors = np.linalg.eig(S)
        sorted_values = np.flip(eigenvalues.argsort(), 0)[:self.pcaComponents]

        self.eigenvalues = eigenvalues[sorted_values]
        self.eigenvectors = eigenvectors[:, sorted_values]
        return self

    def buildGrayLevelModels(self):
        self.meanGrayLevelModels = {}
        self.normalizedGrayLevels = {}
        self.grayLevelsCovariances = {}
        for i, points in enumerate(self.meanLandmark.getPointsAsTuples()):
            # Model gray level appearance
            self.normalizedGrayLevels[i] = []

            # Get gray level model for each landmark and add it
            for landmark in self.landmarks:
                # TODO: rekening houden met tooth?
                grayLevelProfiles, normalizedGrayLevelProfiles = landmark.grayLevelProfileForAllPoints(self.sampleAmount)
                for pointIndex, profile in grayLevelProfiles.items():
                    if pointIndex not in self.meanGrayLevelModels:
                        self.meanGrayLevelModels[pointIndex] = np.zeros(profile.shape)
                    self.meanGrayLevelModels[pointIndex] += profile

                    self.normalizedGrayLevels[i].append(normalizedGrayLevelProfiles[pointIndex])

        for pointIndex, mean in self.meanGrayLevelModels.items():
            self.meanGrayLevelModels[pointIndex] = mean / len(self.meanLandmark.getPointsAsTuples())

            self.grayLevelsCovariances[pointIndex] = np.cov(np.transpose(self.normalizedGrayLevels[pointIndex]))
            #print("COV:",self.grayLevelsCovariances[pointIndex].shape)

        return self

    def MahalanobisDistance(self, profile, landmarkPointIndex):
        Sp = self.grayLevelsCovariances[landmarkPointIndex]
        pMinusMeanTrans = (profile - self.meanGrayLevelModels[landmarkPointIndex]).T
        return pMinusMeanTrans * linalg.inv(Sp) * pMinusMeanTrans

    def matchModelPointsToTargetPoints(self, landmarkY):
        # 1. initialize the shape parameters, b, to zero
        b = np.zeros((self.pcaComponents, 1))
        converged = False

        while not converged:
            # 2. generate the model points using x = x' + Pb
            newModelPoints = self.meanLandmark.points + np.matmul(np.transpose(self.eigenvectors), b).flatten()

            # 3. Find the pose parameters

            # 4. Project Y into the model co-ordinate frame by inverting the transformation T

            # 5. Project y into the tangent plane to x' by scaling: y' = y / (y x')

            # 6. Update the model parameters to match to y'
            newB = np.matmul(self.eigenvectors, yPrime.points - self.meanLandmark.points)
            newB = newB.reshape((self.pcaComponents, -1))
            converged = scipy.spatial.distance.euclidean(b, newB)
            b = newB

        return b

    def reconstruct(self):
        """
        Reconstructs a landmark.
        Be sure to create b for a preprocessed landmark. PCA is done on preprocessed landmarks.
        """
        l = self.preprocessedLandmarks[0]
        b = np.dot(self.eigenvectors.T, l.points - self.meanLandmark.points)
        # b.reshape((self.pcaComponents, -1))
        reconstruction = self.meanLandmark.points + np.dot(self.eigenvectors, b).flatten()

        procrustes_analysis.drawLandmarks([l], "origin")
        procrustes_analysis.drawLandmarks([Landmark(reconstruction)], "reconstruction")
        return reconstruction
