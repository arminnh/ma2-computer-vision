import numpy as np
import scipy.spatial.distance
from sklearn.decomposition import PCA

import procrustes_analysis


class Model:
    def __init__(self, name, landmarks, pcaComponents=20):
        self.name = name
        self.landmarks = landmarks
        self.meanLandmark = None
        self.preprocessedLandmarks = []
        self.pcaComponents = pcaComponents
        self.eigenvectors = []

    def doProcrustesAnalysis(self):
        # procrustes_analysis.drawLandmarks(self.landmarks, "before")

        newLandmarks, meanLandmark = procrustes_analysis.performProcrustesAnalysis(self.landmarks)
        self.preprocessedLandmarks = newLandmarks
        self.meanLandmark = meanLandmark

        procrustes_analysis.drawLandmarks(newLandmarks, "after")

    def doPCA(self):
        data = [l.getPointsAsList() for l in self.landmarks]

        p = PCA(n_components=self.pcaComponents)

        p.fit(data)
        self.eigenvectors = p.components_

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
