import numpy as np
import scipy.spatial.distance
from sklearn.decomposition import PCA

from Landmark import Landmark
import procrustes_analysis


class Model:
    def __init__(self, name, landmarks, pcaComponents=20):
        self.name = name
        self.landmarks = landmarks
        self.meanLandmark = None # type: Landmark
        self.preprocessedLandmarks = []
        self.pcaComponents = pcaComponents
        self.eigenvectors = []
        self.meanTheta = None
        self.meanScale = None

    def doProcrustesAnalysis(self):
        #procrustes_analysis.drawLandmarks(self.landmarks, "before")

        self.preprocessedLandmarks, self.meanLandmark, self.meanTheta, self.meanScale \
            = procrustes_analysis.performProcrustesAnalysis(self.landmarks)

        #procrustes_analysis.drawLandmarks(self.preprocessedLandmarks, "after")

    def translateMean(self, x, y):
        self.meanLandmark = self.meanLandmark.translatePoints(x,y)
        return self.meanLandmark

    def translateAndRescaleMean(self, x, y):
        self.meanLandmark = self.meanLandmark.normalize()
        self.meanLandmark.points *= self.meanScale
        mean = self.translateMean(x,y)
        return mean

    def doPCA(self):
        data = [l.getPointsAsList() for l in self.preprocessedLandmarks]

        #[m, eigenvalues, self.eigenvectors] = self._pca(data,self.pcaComponents)


        S = np.cov(np.transpose(data))

        eigenvalues, eigenvectors = np.linalg.eig(S)
        sorted_values = np.flip(eigenvalues.argsort(), 0)[:self.pcaComponents]
        eigvals = eigenvalues[sorted_values]
        eigvectors = eigenvectors[:, sorted_values]

        self.eigenvectors = eigvectors

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

    def reconstruct(self):
        l = self.preprocessedLandmarks[0]
        b = np.dot(self.eigenvectors.T, l.points - self.meanLandmark.points)
        #b.reshape((self.pcaComponents, -1))
        recon = self.meanLandmark.points + np.dot(self.eigenvectors, b).flatten()

        procrustes_analysis.drawLandmarks([l], "origin")
        procrustes_analysis.drawLandmarks([Landmark(recon)], "recon")