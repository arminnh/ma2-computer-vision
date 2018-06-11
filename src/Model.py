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


    def _covariance(self, data):
        mean = self.meanLandmark.getPointsAsList()

        S = np.matmul((data-mean).T,(data- mean))
        print(S/ (len(data)-1))
        print("numpy")
        print(np.cov(np.transpose(data)))
        return S / len(data)

    def getT(self, data):
        mean = self.meanLandmark.getPointsAsList()
        mean = np.asarray(mean)
        D = data - mean
        S = np.matmul(D, D.T)
        eigenvalues, eigenvectors = np.linalg.eig(S)
        sorted_values = np.flip(eigenvalues.argsort(), 0)[:len(data)]
        eigvals = eigenvalues[sorted_values]
        eigvectors = eigenvectors[:, sorted_values]
        return np.matmul(D.T, eigvectors)


    def _pca(self, X, number_of_components):

        mean = np.mean(X, 0)
        X_mean = X - mean

        M = np.dot(X_mean, X_mean.T)
        print(M.shape)
        eigenvalues, eigenvectors = np.linalg.eig(M)
        #eigenvectors = np.dot(X_mean.T, eigenvectors)

        # Get the most important eigenvectors first
        sorted_values = np.flip(eigenvalues.argsort(), 0)[:number_of_components]

        return [mean, eigenvalues[sorted_values], eigenvectors[:, sorted_values]]


    def doPCA(self):
        data = [l.getPointsAsList() for l in self.preprocessedLandmarks]

        #[m, eigenvalues, self.eigenvectors] = self._pca(data,self.pcaComponents)

        p = PCA(n_components=self.pcaComponents)

        S = np.cov(np.transpose(data))

        eigenvalues, eigenvectors = np.linalg.eig(S)
        sorted_values = np.flip(eigenvalues.argsort(), 0)[:self.pcaComponents]
        eigvals = eigenvalues[sorted_values]
        eigvectors = eigenvectors[:, sorted_values]
        print("shape:",eigvectors.shape)

        T = self.getT(data)


        #print(S.shape)
        #covariance = self._covariance(data)
        #p.fit(data)
        self.eigenvectors = T

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
        l = self.landmarks[0]
        print(self.eigenvectors.shape)
        b = np.dot(self.eigenvectors.T, l.points - self.meanLandmark.points)
        #b.reshape((self.pcaComponents, -1))
        recon = self.meanLandmark.points + np.dot(self.eigenvectors, b).flatten()

        procrustes_analysis.drawLandmarks([l], "origin")
        procrustes_analysis.drawLandmarks([Landmark(recon)], "recon")
