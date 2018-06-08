from typing import Type

import numpy as np
import math

class Landmark:

    def __init__(self, fileName, id):
        self.id = id

        # This is a list of point coordinates in the form [x_0, y_0, x_1, y_1
        self.points = self._loadPoints(fileName)

    def _loadPoints(self, fileName):
        f = open(fileName, "r")
        p = f.readlines()
        return np.asarray([float(x) for x in p])
        #return [(float(p[2*j]),float(p[2*j+1])) for j in range(len(p)/2)]

    def getPointsAsTuples(self):
        p = self.points
        return np.asarray([(float(p[2*j]),float(p[2*j+1])) for j in range(int(len(p)/2))])

    def getPointsAsList(self):
        return self.points

    def getTranslatedPoints(self):
        """
        Get's the translation of this shape
        The translation is the mean x and mean y
        :return:
        """
        # https://en.wikipedia.org/wiki/Procrustes_analysis
        t = np.mean(self.getPointsAsTuples(), axis=0)
        return self.getPointsAsTuples() - t

    def getScale(self):
        """
        Get's the scale of the shape
        :return:
        """
        # https://en.wikipedia.org/wiki/Procrustes_analysis
        points = self.getTranslatedPoints()

        s = np.sqrt(np.sum(np.square(points))/float(len(points)))
        return s

    def getPointsWithoutScaleAndTranslation(self):
        points = self.getTranslatedPoints()
        s = self.getScale()
        return (points) / s

    def standardize(self):
        p = self.getPointsWithoutScaleAndTranslation()
        self.points = p.flatten()

    def getThetaForReference(self, ref_p):
        """
        Superimpose
        :param other:
        :return:
        """
        current_p = self.getPointsAsTuples()

        s1 = 0
        s2 = 0
        for i, p in enumerate(ref_p):

            #     w_i * y_i - z_i * x_i
            s1 += current_p[i][0] * p[1]  - current_p[i][1] * p[0]
            s2 += p[0]* current_p[i][0]  + p[1]*current_p[i][1]
        theta = math.atan(s1/s2)

        return theta

    def rotate(self, theta):
        new_points = []
        for p in self.getPointsAsTuples():
            u = math.cos(theta) * p[0] - math.sin(theta) * p[1]
            v = math.sin(theta) * p[0] + math.cos(theta) * p[1]

            new_points.append(u)
            new_points.append(v)

        self.points = np.asarray(new_points)

    def getDistance(self, ref):
        return np.sqrt(np.sum(np.square(self.getPointsAsList() - ref)))

    # def procustesWithLandmark(self,other):
    #     otherM, newP, disp = procrustes(other.getPointsAsTuples(), self.getPointsAsTuples())
    #     print(otherM.flatten())
