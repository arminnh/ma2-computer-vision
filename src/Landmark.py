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
        return [float(x) for x in p]
        #return [(float(p[2*j]),float(p[2*j+1])) for j in range(len(p)/2)]

    def getPointsAsTuples(self):
        p = self.points
        return [(float(p[2*j]),float(p[2*j+1])) for j in range(int(len(p)/2))]

    def getPointsAsList(self):
        return self.points

    def getTranslation(self):
        """
        Get's the translation of this shape
        The translation is the mean x and mean y
        :return:
        """
        # https://en.wikipedia.org/wiki/Procrustes_analysis
        return np.mean(self.getPointsAsTuples(), axis=0)


    def getScale(self):
        """
        Get's the scale of the shape
        :return:
        """
        # https://en.wikipedia.org/wiki/Procrustes_analysis
        t = self.getTranslation()
        points = self.getPointsAsTuples()

        s = np.sqrt(np.sum(np.square(points - t))/len(points))
        return s

    def getPointsWithoutScaleAndTranslation(self):
        points = self.getPointsAsTuples()
        t = self.getTranslation()
        s = self.getScale()
        return (points - t) / s


    def superImpose(self, other):
        """
        Superimpose
        :param other:
        :return:
        """
        other_p = other.getPointsWithoutScaleAndTranslation()
        current_p = self.getPointsWithoutScaleAndTranslation()
        s1 = 0
        s2 = 0

        for i, p in enumerate(other_p):

            #     w_i * y_i - z_i * x_i
            s1 += current_p[i][0] * p[1]  - current_p[i][1] * p[0]
            s2 += p[0]* current_p[i][0]  + p[1]*current_p[i][1]
        theta = math.atan(s1/s2)

        UV = []
        for p in current_p:
            u = math.cos(theta) * p[0] - math.sin(theta) * p[1]
            v = math.sin(theta) * p[0] + math.cos(theta) * p[1]

            UV.append(u)
            UV.append(v)

        self.points = UV

