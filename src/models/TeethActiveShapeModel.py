import numpy as np
import scipy.spatial.distance
from scipy import linalg

import procrustes_analysis
from Landmark import Landmark


class TeethActiveShapeModel:
    """ Data structure for Multi-resolution Active Shape Model search """

    def __init__(self, individualLandmarks, setLandmarks, resolutionLevels, maxLevelIterations, grayLevelModelSize,
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
        self.resolutionLevels = resolutionLevels
        self.maxLevelIterations = maxLevelIterations
        self.grayLevelModelSize = grayLevelModelSize
        self.sampleAmount = sampleAmount
        self.pClose = pClose
        self.pcaComponents = pcaComponents
        self.eigenvalues = {}
        self.eigenvectors = {}
        self.grayLevelModelPyramid = {}

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

    def getTranslatedAndInverseScaledMeanMouth(self, resolutionLevel, x, y):
        """ Returns the mean landmark rescaled back from unit variance (after procrustes) and translated to x and y. """
        return self.meanMouthLandmark.scale(self.meanMouthScale*0.85).scale(0.5 ** resolutionLevel).translate(x, y)

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

    def getShapeParametersForToothLandmark(self, toothNumber, landmark):
        b = self.eigenvectors[toothNumber].T @ (landmark.points - self.meanLandmarks[toothNumber].points)
        return b.reshape((self.pcaComponents, -1))

    def reconstructToothLandmarkForShapeParameters(self, toothNumber, b):
        return Landmark(self.meanLandmarks[toothNumber].points + (self.eigenvectors[toothNumber] @ b).flatten())

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
        mouthLandmark = Landmark(points=np.asarray([]))

        for toothNumber in range(1, 9):
            toothLandmarkY = Landmark(points=mouthLandmarkY.points[(toothNumber - 1) * 80:toothNumber * 80])

            b = np.zeros((self.pcaComponents, 1))
            diff = float("inf")
            translateX = 0
            translateY = 0
            theta = 0
            scale = 0

            while diff > 1e-9:
                # Generate model points using x = x' + Pb
                x = self.reconstructToothLandmarkForShapeParameters(toothNumber, b)

                # Project Y into the model coordinate frame by superimposition and fetch the pose parameters
                y, (translateX, translateY), scale, theta = toothLandmarkY.superimpose(x)

                # Update the shape parameters b
                newB = self.getShapeParametersForToothLandmark(toothNumber, y)

                # Apply constraints to the shape parameters b to ensure plausible shapes
                toothEigenvalues = self.eigenvalues[toothNumber]
                for i in range(len(newB)):
                    limit = 3 * np.sqrt(abs(toothEigenvalues[i]))
                    newB[i] = np.clip(newB[i], -limit, limit)

                diff = scipy.spatial.distance.euclidean(b, newB)
                b = newB

            reconstructedTooth = self.reconstructToothLandmarkForShapeParameters(toothNumber, b) \
                .rotate(-theta).scale(scale).translate(-translateX, -translateY)

            mouthLandmark.points = np.concatenate((mouthLandmark.points, reconstructedTooth.points))

        return mouthLandmark

    def reconstruct(self, landmarks):
        """
        Reconstructs a landmark.
        Be sure to create b for a preprocessed landmark. PCA is done on preprocessed landmarks.
        """
        procrustes_analysis.plotLandmarks(landmarks.values(), "input landmarks")

        reconstructions = []

        for toothNumber, landmark in landmarks.items():
            superimposed, (translateX, translateY), scale, theta = landmark.superimpose(self.meanLandmarks[toothNumber])

            b = self.getShapeParametersForToothLandmark(toothNumber, superimposed)
            reconstructed = self.reconstructToothLandmarkForShapeParameters(toothNumber, b)
            print("shape b = {}, shape eigenvectors = {}".format(b.shape, self.eigenvectors[toothNumber].shape))

            reconstructions.append(reconstructed.rotate(-theta).scale(scale).translate(-translateX, -translateY))

        procrustes_analysis.plotLandmarks(
            reconstructions,
            "reconstructed landmarks, PCA components = {}".format(self.pcaComponents)
        )
        return reconstructions
