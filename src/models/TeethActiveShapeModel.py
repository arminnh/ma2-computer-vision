import numpy as np
import scipy.spatial.distance
from scipy import linalg

import procrustes_analysis
from Landmark import Landmark


class TeethActiveShapeModel:
    """ Data structure for Multi-resolution Active Shape Model search """

    def __init__(self, individualLandmarks, setLandmarks, maxResolutionLevel, maxLevelIterations, grayLevelModelSize,
                 sampleAmount, pClose, pcaComponents):
        # dict of toothnumber -> list landmarks for tooth
        self.toothLandmarks = individualLandmarks
        # list of landmarks that contain all 8 teeth
        self.mouthLandmarks = setLandmarks
        self.preprocessedLandmarks = {}
        self.meanLandmarks = {}
        self.meanThetas = {}
        self.meanScales = {}
        self.meanMouthLandmark = None
        self.meanMouthTheta = 0
        self.meanMouthScale = 1
        self.maxResolutionLevel = maxResolutionLevel
        self.maxLevelIterations = maxLevelIterations
        self.grayLevelModelSize = grayLevelModelSize
        self.sampleAmount = sampleAmount
        self.pClose = pClose
        self.pcaComponents = pcaComponents
        self.eigenvalues = {}
        self.eigenvectors = {}
        self.meanProfilesForLandmarkPoints = {}
        self.grayLevelModelCovarianceMatrices = {}

    def doProcrustesAnalysis(self):
        for toothNumber, landmarks in self.toothLandmarks.items():
            preprocessedLandmarks, meanLandmark, meanScale, meanTheta \
                = procrustes_analysis.performProcrustesAnalysis(landmarks)

            self.preprocessedLandmarks[toothNumber] = preprocessedLandmarks
            self.meanLandmarks[toothNumber] = meanLandmark
            self.meanScales[toothNumber] = meanScale
            self.meanThetas[toothNumber] = meanTheta

        _, self.meanMouthLandmark, self.meanMouthScale, self.meanMouthTheta \
            = procrustes_analysis.performProcrustesAnalysis(self.mouthLandmarks)

        return self

    def getTranslatedAndInverseScaledMeanMouth(self, x, y):
        """ Returns the mean landmark rescaled back from unit variance (after procrustes) and translated to x and y. """
        return self.meanMouthLandmark.scale(self.meanMouthScale).translate(x, y)

    def doPCA(self):
        """ Perform PCA on the landmarks after procrustes analysis and store the eigenvalues and eigenvectors. """
        for toothNumber, landmarks in self.preprocessedLandmarks.items():
            data = [l.points for l in landmarks]
            data.append(data[0])

            S = np.cov(np.transpose(data))

            eigenvalues, eigenvectors = np.linalg.eig(S)
            sorted_values = np.flip(eigenvalues.argsort(), 0)[:self.pcaComponents]

            self.eigenvalues[toothNumber] = eigenvalues[sorted_values]
            self.eigenvectors[toothNumber] = eigenvectors[:, sorted_values]

        return self

    def getShapeParametersForLandmark(self, toothNumber, landmark):
        b = self.eigenvectors[toothNumber].T @ (landmark.points - self.meanLandmarks[toothNumber].points)
        return b.reshape((self.pcaComponents, -1))

    def reconstructLandmarkForShapeParameters(self, toothNumber, b):
        return Landmark(self.meanLandmarks[toothNumber].points + (self.eigenvectors[toothNumber] @ b).flatten())

    def buildGrayLevelModels(self):
        """
        Builds gray level models for each of the tooth's landmark points.
        Build gray level models for each of the mean landmark points by averaging the gray level profiles for each
        point of each landmark.
        """
        self.grayLevelModelCovarianceMatrices = {}
        self.meanProfilesForLandmarkPoints = {}

        # Build gray level model for each landmark
        for i, landmark in enumerate(self.mouthLandmarks):
            # Get the gray level profiles for each of the 40 landmark points
            normalizedGrayLevelProfiles = landmark.normalizedGrayLevelProfilesForLandmarkPoints(
                img=landmark.radiograph.img,
                grayLevelModelSize=self.grayLevelModelSize
            )

            for j, normalizedProfile in normalizedGrayLevelProfiles.items():
                if j not in self.meanProfilesForLandmarkPoints:
                    self.meanProfilesForLandmarkPoints[j] = []

                self.meanProfilesForLandmarkPoints[j].append(normalizedProfile)

        for pointIdx in range(len(self.meanProfilesForLandmarkPoints)):
            cov = np.cov(np.transpose(self.meanProfilesForLandmarkPoints[pointIdx]))
            self.grayLevelModelCovarianceMatrices[pointIdx] = linalg.pinv(cov)

            # Replace each point's list of gray level profiles by their means
            self.meanProfilesForLandmarkPoints[pointIdx] = np.mean(self.meanProfilesForLandmarkPoints[pointIdx], axis=0)

        return self

    def mahalanobisDistance(self, normalizedGrayLevelProfile, landmarkPointIndex):
        """
        Returns the squared Mahalanobis distance of a new gray level profile from the built gray level model with index
        landmarkPointIndex.
        """
        Sp = self.grayLevelModelCovarianceMatrices[landmarkPointIndex]
        pMinusMeanTrans = (normalizedGrayLevelProfile - self.meanProfilesForLandmarkPoints[landmarkPointIndex])

        return pMinusMeanTrans.T @ Sp @ pMinusMeanTrans

    def findBetterFittingLandmark(self, img, landmark):
        """
        Active Shape Model Algorithm: An iterative approach to improving the fit of an instance X.
        Returns a landmark that is a better fit on the image than the given according to the gray level pointProfiles of
        points of the landmark and the mahalanobis distance.
        """
        # Examine a region of the image around each point X_i to find the best nearby match for the point X_i'
        # Get the gray level pointProfiles of points on normal lines of the landmark's points
        profilesForLandmarkPoints = landmark.getGrayLevelProfilesForNormalPoints(
            img=img,
            sampleAmount=self.sampleAmount,
            grayLevelModelSize=self.grayLevelModelSize,
            derive=True
        )

        bestPoints = []

        # landmarkPointIdx = the points 0 to 39 on the landmark
        for landmarkPointIdx in range(len(profilesForLandmarkPoints)):
            # landmarkPointProfiles = list of {grayLevelProfile, normalPoint, grayLevelProfilePoints}
            landmarkPointProfiles = profilesForLandmarkPoints[landmarkPointIdx]
            distances = []

            for profileContainer in landmarkPointProfiles:
                grayLevelProfile = profileContainer["grayLevelProfile"]
                normalPoint = profileContainer["normalPoint"]

                d = self.mahalanobisDistance(grayLevelProfile, landmarkPointIdx)
                distances.append((abs(d), normalPoint))
                print("Mahalanobis dist: {:.2f}, p: {}".format(abs(d), normalPoint))

            bestPoints.append(min(distances, key=lambda x: x[0])[1])

        landmark = landmark.copy(np.asarray(bestPoints).flatten())

        # Find the pose parameters that best fit the new found points X
        landmark, (translateX, translateY), scale, theta = landmark.superimpose(self.meanLandmarks)

        # Apply constraints to the parameters b to ensure plausible shapes
        b = self.getShapeParametersForLandmark(landmark)

        # Constrain the shape parameters to lie within certain limits
        for i in range(len(b)):
            limit = 2 * np.sqrt(abs(self.eigenvalues[i]))
            b[i] = np.clip(b[i], -limit, limit)

        return self.reconstructLandmarkForShapeParameters(b) \
            .rotate(-theta).scale(scale).translate(-translateX, -translateY)

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
            x = self.reconstructLandmarkForShapeParameters(b)

            # Project Y into the model coordinate frame by superimposition
            # and get the parameters of the transformation
            y, (translateX, translateY), scale, theta = landmarkY.superimpose(x)

            # Update the model parameters b
            newB = self.getShapeParametersForLandmark(y)

            diff = scipy.spatial.distance.euclidean(b, newB)
            b = newB

        return self.reconstructLandmarkForShapeParameters(b) \
            .rotate(-theta).scale(scale).translate(-translateX, -translateY)

    def reconstruct(self, landmarks):
        """
        Reconstructs a landmark.
        Be sure to create b for a preprocessed landmark. PCA is done on preprocessed landmarks.
        """
        procrustes_analysis.plotLandmarks(landmarks.values(), "input landmarks")

        reconstructions = []

        for toothNumber, landmark in landmarks.items():
            superimposed, (translateX, translateY), scale, theta = landmark.superimpose(self.meanLandmarks[toothNumber])

            b = self.getShapeParametersForLandmark(toothNumber, superimposed)
            reconstructed = self.reconstructLandmarkForShapeParameters(toothNumber, b)
            print("shape b = {}, shape eigenvectors = {}".format(b.shape, self.eigenvectors[toothNumber].shape))

            reconstructions.append(reconstructed.rotate(-theta).scale(scale).translate(-translateX, -translateY))

        procrustes_analysis.plotLandmarks(
            reconstructions,
            "reconstructed landmarks, PCA components = {}".format(self.pcaComponents)
        )
        return reconstructions
