import numpy as np
import scipy
from scipy import linalg

import procrustes_analysis
from Landmark import Landmark
class initModel:

    def __init__(self, name, landmarks, rnge, pcaComponents, sampleAmount):
        # Is a list of lists
        # each inner list contains four incisors either from upper or lower jaw
        # all must be from the same jaw though
        self.name = name
        self.originalLandmarks = landmarks
        self.crownLandmarks = self._getToothCrowns(landmarks, rnge)
        self.sampleAmount = 20
        self.preprocessedLandmarks = []
        self.meanLandmark = None  # type: Landmark
        self.meanTheta = None
        self.meanScale = None
        self.pcaComponents = pcaComponents
        self.eigenvalues = np.array([])
        self.eigenvectors = np.array([])
        self.sampleAmount = sampleAmount
        self.y_ij = {}
        self.meanProfilesForLandmarkPoints = {}
        self.C_yj = {}

    def _getToothCrowns(self, landmarksList,rnge):
        newLandmarks = []
        for lst in landmarksList:
            crownPoints = []
            toothNmbr = 0

            for landmark in lst:
                toothNmbr = landmark.toothNumber
                crownPoints = crownPoints + list(landmark.getPointsAsTuples()[rnge].flatten())

            if self.name == -1:
                toothNmbr = -1

            newLandmark = Landmark(crownPoints,toothNumber=toothNmbr)
            newLandmark.radiograph = lst[0].radiograph
            newLandmarks.append(newLandmark)

        return newLandmarks

    def plotLandmarks(self):
        import matplotlib.pyplot as plt
        plt.figure()

        for landmark in self.crownLandmarks[:2]:
            # PIL can't work with numpy arrays so convert to list of tuples
            points = landmark.getPointsAsTuples()
            X = points[:, 0]
            Y = points[:, 1]
            plt.plot(X, Y, 'x')
            for i in range(len(points)):
                plt.text(X[i] - 10, Y[i], i)

        plt.legend()
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.axis("equal")
        plt.show()

    def doProcrustesAnalysis(self):
        # procrustes_analysis.drawLandmarks(self.landmarks, "before")

        self.preprocessedLandmarks, self.meanLandmark, self.meanScale, self.meanTheta \
            = procrustes_analysis.performProcrustesAnalysis(self.crownLandmarks)

        #procrustes_analysis.plotLandmarks([self.meanLandmark], "mean")
        #procrustes_analysis.plotLandmarks(self.preprocessedLandmarks, "after")
        return self

    def getTranslatedMean(self, x, y):
        """ Returns the mean landmark translated to x and y. """
        return self.meanLandmark.translate(x, y)

    def getTranslatedAndInverseScaledMean(self, x, y):
        """ Returns the mean landmark rescaled back from unit variance (after procrustes) and translated to x and y. """
        return self.meanLandmark.scale(self.meanScale * 0.75).translate(x, y)

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
        self.C_yj = {}
        self.meanProfilesForLandmarkPoints = {}

        # Build gray level model for each landmark
        for i, landmark in enumerate(self.crownLandmarks):
            # Get the gray level profiles for each of the 40 landmark points
            normalizedGrayLevelProfiles = landmark.normalizedGrayLevelProfilesForLandmarkPoints(
                img=landmark.getCorrectRadiographPart(),
                sampleAmount=self.sampleAmount
            )

            for j, normalizedProfile in normalizedGrayLevelProfiles.items():
                if j not in self.meanProfilesForLandmarkPoints:
                    self.meanProfilesForLandmarkPoints[j] = []

                self.meanProfilesForLandmarkPoints[j].append(normalizedProfile)

        for pointIdx in range(len(self.meanProfilesForLandmarkPoints)):
            cov = np.cov(np.transpose(self.meanProfilesForLandmarkPoints[pointIdx]))
            self.C_yj[pointIdx] = linalg.pinv(cov)

            # Replace each point's list of gray level profiles by their means
            self.meanProfilesForLandmarkPoints[pointIdx] = np.mean(self.meanProfilesForLandmarkPoints[pointIdx], axis=0)

        return self

    def mahalanobisDistance(self, normalizedGrayLevelProfile, landmarkPointIndex):
        """
        Returns the squared Mahalanobis distance of a new gray level profile from the built gray level model with index
        landmarkPointIndex.
        """
        Sp = self.C_yj[landmarkPointIndex]
        pMinusMeanTrans = (normalizedGrayLevelProfile - self.meanProfilesForLandmarkPoints[landmarkPointIndex])

        return pMinusMeanTrans.T @ Sp @ pMinusMeanTrans

    def findBetterFittingLandmark(self, landmark, img):
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
        landmark, (translateX, translateY), scale, theta = landmark.superimpose(self.meanLandmark)

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

    def reconstruct(self):
        """
        Reconstructs a landmark.
        Be sure to create b for a preprocessed landmark. PCA is done on preprocessed landmarks.
        """
        landmark = self.preprocessedLandmarks[0]
        b = self.getShapeParametersForLandmark(landmark)

        reconstructed = self.reconstructLandmarkForShapeParameters(b)

        procrustes_analysis.plotLandmarks([landmark], "origin")
        procrustes_analysis.plotLandmarks([reconstructed], "reconstructed")
        return reconstructed
