import numpy as np
import scipy.spatial.distance
from scipy import linalg

import procrustes_analysis
from Landmark import Landmark


class TeethActiveShapeModel:
    """ Data structure for Multi-resolution Active Shape Model search """

    def __init__(self, mouthLandmarks, resolutionLevels, maxLevelIterations, grayLevelModelSize,
                 sampleAmount, pClose, pcaComponents):

        self.mouthLandmarks = mouthLandmarks  # list of landmarks that contain all 8 teeth
        self.preprocessedLandmarks = []
        self.meanLandmark = None
        self.meanTheta = 0
        self.meanScale = 1
        self.resolutionLevels = resolutionLevels
        self.maxLevelIterations = maxLevelIterations
        self.grayLevelModelSize = grayLevelModelSize
        self.sampleAmount = sampleAmount
        self.pClose = pClose
        self.pcaComponents = pcaComponents
        self.eigenvalues = np.asarray([])
        self.eigenvectors = np.asarray([])
        self.grayLevelModelPyramid = {}

    def doProcrustesAnalysis(self):
        self.preprocessedLandmarks, self.meanLandmark, self.meanScale, self.meanTheta \
            = procrustes_analysis.performProcrustesAnalysis(self.mouthLandmarks)

        return self

    def getTranslatedAndInverseScaledMeanMouth(self, resolutionLevel, x, y):
        """ Returns the mean landmark rescaled back from unit variance (after procrustes) and translated to x and y. """
        return self.meanLandmark.scale(self.meanScale).scale(0.5 ** resolutionLevel).translate(x, y)

    def doPCA(self):
        """ Perform PCA on the landmarks after procrustes analysis and store the eigenvalues and eigenvectors. """
        data = [l.points for l in self.preprocessedLandmarks]

        S = np.cov(np.transpose(data))

        eigenvalues, eigenvectors = np.linalg.eig(S)
        sorted_values = np.flip(eigenvalues.argsort(), 0)[:self.pcaComponents]

        self.eigenvalues = eigenvalues[sorted_values]
        self.eigenvectors = eigenvectors[:, sorted_values]

        return self

    def getShapeParametersForLandmark(self, landmark):
        b = self.eigenvectors.T @ (landmark.points - self.meanLandmark.points)
        return b.reshape((self.pcaComponents, -1))

    def reconstructLandmarkForShapeParameters(self, b):
        return Landmark(self.meanLandmark.points + (self.eigenvectors @ b).flatten())

    def buildGrayLevelModels(self):
        """
        Builds gray level models for each of the tooth's landmark points.
        Build gray level models for each of the mean landmark points by averaging the gray level profiles for each
        point of each landmark.
        """
        self.grayLevelModelPyramid = {}

        # Build a gray level model for each resolution level
        for resolutionLevel in range(self.resolutionLevels):
            profilesForLandmarkPoints = {}
            meanProfilesForLandmarkPoints = {}
            covarianceForLandmarkPoints = {}

            # Scale the landmarks down so that their coordinates fit in the image at this resolution level
            scaledLandmarks = [l.scale(0.5 ** resolutionLevel) for l in self.mouthLandmarks]

            # Build gray level model for each landmark
            for i, landmark in enumerate(scaledLandmarks):
                # Get the gray level profiles for each of the 40 landmark points
                normalizedGrayLevelProfiles = landmark.normalizedGrayLevelProfilesForLandmarkPoints(
                    img=landmark.radiograph.imgPyramid[resolutionLevel],
                    grayLevelModelSize=self.grayLevelModelSize
                )

                for j, normalizedProfile in normalizedGrayLevelProfiles.items():
                    if j not in profilesForLandmarkPoints:
                        profilesForLandmarkPoints[j] = []

                    profilesForLandmarkPoints[j].append(normalizedProfile)

            # Store the mean and covariance matrix of each landmark point's gray level profiles
            for pointIdx in range(len(profilesForLandmarkPoints)):
                meanProfilesForLandmarkPoints[pointIdx] = np.mean(profilesForLandmarkPoints[pointIdx], axis=0)

                cov = np.cov(np.transpose(profilesForLandmarkPoints[pointIdx]))
                covarianceForLandmarkPoints[pointIdx] = linalg.pinv(cov)

            self.grayLevelModelPyramid[resolutionLevel] = {
                "meanProfilesForLandmarkPoints": meanProfilesForLandmarkPoints,
                "covarianceForLandmarkPoints": covarianceForLandmarkPoints
            }

        return self

    def mahalanobisDistance(self, resolutionLevel, landmarkPointIndex, normalizedGrayLevelProfile):
        """
        Returns the squared Mahalanobis distance of a new gray level profile from the built gray level model with index
        landmarkPointIndex.
        """
        Sp = self.grayLevelModelPyramid[resolutionLevel]["covarianceForLandmarkPoints"][landmarkPointIndex]
        meanProfile = self.grayLevelModelPyramid[resolutionLevel]["meanProfilesForLandmarkPoints"][landmarkPointIndex]

        pMinusMeanTrans = (normalizedGrayLevelProfile - meanProfile)

        return pMinusMeanTrans.T @ Sp @ pMinusMeanTrans

    def findBetterFittingLandmark(self, resolutionLevel, img, landmark):
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
            # landmarkPointProfiles = list of {normalPoint, grayLevelProfile, grayLevelProfilePoints}
            landmarkPointProfiles = profilesForLandmarkPoints[landmarkPointIdx]
            distances = []

            for profileContainer in landmarkPointProfiles:
                grayLevelProfile = profileContainer["grayLevelProfile"]
                normalPoint = profileContainer["normalPoint"]

                d = self.mahalanobisDistance(resolutionLevel, landmarkPointIdx, grayLevelProfile)
                distances.append((abs(d), normalPoint))

            bestPoints.append(min(distances, key=lambda x: x[0])[1])

        return landmark.copy(np.asarray(bestPoints).flatten())

    def matchModelPointsToTargetPoints(self, mouthLandmarkY):
        """
        A simple iterative approach towards finding the best pose and shape parameters to match a model instance X to a
        new set of image points Y.
        """
        b = np.zeros((self.pcaComponents, 1))
        diff = 1
        translateX = 0
        translateY = 0
        theta = 0
        scale = 0

        while diff > 1e-9:
            # Generate model points using x = x' + Pb
            x = self.reconstructLandmarkForShapeParameters(b)

            # Project Y into the model coordinate frame by superimposition and fetch the pose parameters
            y, (translateX, translateY), scale, theta = mouthLandmarkY.superimpose(x)

            # Update the shape parameters b
            newB = self.getShapeParametersForLandmark(y)

            # Apply constraints to the shape parameters b to ensure plausible shapes
            for i in range(len(newB)):
                limit = 2 * np.sqrt(abs(self.eigenvalues[i]))
                newB[i] = np.clip(newB[i], -limit, limit)

            diff = scipy.spatial.distance.euclidean(b, newB)
            b = newB

        return self.reconstructLandmarkForShapeParameters(b) \
            .rotate(-theta).scale(scale).translate(-translateX, -translateY)

    def reconstruct(self, mouthLandmark):
        """
        Reconstructs a landmark.
        Be sure to create b for a preprocessed landmark. PCA is done on preprocessed landmarks.
        """
        superimposed, (translateX, translateY), scale, theta = mouthLandmark.superimpose(self.meanLandmark)

        b = self.getShapeParametersForLandmark(superimposed)
        reconstructed = self.reconstructLandmarkForShapeParameters(b)
        reconstructed = reconstructed.rotate(-theta).scale(scale).translate(-translateX, -translateY)

        procrustes_analysis.plotLandmarks(
            [mouthLandmark, reconstructed],
            "original + reconstructed landmark, PCA components = {}".format(self.pcaComponents)
        )
        print("shape b = {}, shape eigenvectors = {}".format(b.shape, self.eigenvectors.shape))
        return reconstructed
