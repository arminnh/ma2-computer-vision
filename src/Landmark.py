import math
import os
import re

import numpy as np

import util


class Landmark:

    def __init__(self, points, radiographFilename=None, toothNumber=None):
        self.radiographFilename = radiographFilename
        self.toothNumber = toothNumber
        self.points = points if isinstance(points, np.ndarray) else np.array(points)
        self.radiograph = None

    def __str__(self):
        return "Landmark for tooth {} for radiograph {}".format(self.toothNumber, self.radiographFilename)

    def translatePoints(self, x, y):
        p = self.points.copy()
        p[0::2] = p[0::2] + x
        p[1::2] = p[1::2] + y

        return Landmark(p, self.radiographFilename, self.toothNumber)

    def getPointsAsTuples(self):
        p = list(self.points)
        # [ [x_0, y_0], [x_1, y_1] ... ]
        return np.asarray([(float(p[2 * j]), float(p[2 * j + 1])) for j in range(int(len(p) / 2))])

    def getPointsAsList(self):
        return self.points

    def getMeanShiftedPoints(self):
        """ Returns the landmark points translated by their means. """
        # https://en.wikipedia.org/wiki/Procrustes_analysis
        p = self.getPointsAsTuples()
        t = np.mean(p, axis=0)
        return p - t

    def getScale(self):
        """ Returns a statistical measure of the object's scale, root mean square distance (RMSD). """
        # https://en.wikipedia.org/wiki/Procrustes_analysis
        # TODO: check if this scaling factor is correct, maybe scale using SVD
        distance = self.getMeanShiftedPoints()
        return np.sqrt(np.mean(np.square(distance)))

    def getNormalizedPoints(self):
        """ Returns an array of normalized landmark points.  """
        return self.getMeanShiftedPoints() / self.getScale()

    def getThetaForReference(self, reference):
        """
        Superimpose
        :param reference:
        :type reference: Landmark
        """
        current_p = self.getNormalizedPoints()

        s1 = 0
        s2 = 0
        for i, p in enumerate(reference.getNormalizedPoints()):
            #     w_i * y_i - z_i * x_i
            s1 += current_p[i][0] * p[1] - current_p[i][1] * p[0]
            s2 += p[0] * current_p[i][0] + p[1] * current_p[i][1]

        theta = math.atan(s1 / s2)

        return theta

    def shapeDistance(self, other):
        """ Returns the SSD from an other landmark.
        Should be done on superimposed (translated, scaled, and rotated) objects.
        :type other: Landmark
        """
        return np.sqrt(np.sum(np.square(self.getPointsAsList() - other.getPointsAsList())))

    def normalize(self):
        """ Normalizes the points in the landmark. """
        return Landmark(self.getNormalizedPoints().flatten(), self.radiographFilename, self.toothNumber)

    def rotate(self, theta):
        """ Rotates the points in the landmark. """
        new_points = []
        for p in self.getNormalizedPoints():
            u = math.cos(theta) * p[0] - math.sin(theta) * p[1]
            v = math.sin(theta) * p[0] + math.cos(theta) * p[1]

            new_points.append(u)
            new_points.append(v)

        return Landmark(np.asarray(new_points), self.radiographFilename, self.toothNumber)

    def superimpose(self, other):
        """ Returns this landmark superimposed (translated, scaled, and rotated) over another. """
        theta = self.getThetaForReference(other)
        s = self.getScale()
        superimposed = self.normalize().rotate(theta)

        return Landmark(superimposed.points, self.radiographFilename, self.toothNumber), theta, s

    def grayLevelProfileForAllPoints(self, pixelsToSample):
        """
        For every landmark point j (all points in this landmark) in the image i (the radiograph of this landmark) of
        the training set, we extract a gray level profile g_ij of length n_p pixels, centered around the landmark point.
        Not the actual gray level profile but its normalized derivative to get invariance to the offsets and uniform
        scaling of the gray level.
        The gray level profile of a landmark point j is a vector of n_p values.
        ~ "Active Shape Models - Part I: Modeling Shape and Gray Level Variations"
        """
        if self.radiograph is None:
            raise Exception("Need radiograph for gray level profile")

        grayLevelProfiles = {}

        points = self.getPointsAsTuples()
        for i, point in enumerate(points):
            # Build gray level profile by sampling a few points on each side of the point.

            # Sample points on normal line of the current landmark point
            pointsOnNormalLine = util.sampleNormalLine(points[i - 1], point, points[(i + 1) % len(points)],
                                                       pixelsToSample=pixelsToSample)

            # Get pixel values on the sampled positions


            # Derivative profile of length n_p - 1

            # Normalized derivative profile
            grayLevelProfiles[i] = profile

        return grayLevelProfiles


def loadLandmarkPoints(filename):
    f = open(filename, "r")
    p = f.readlines()
    return np.asarray([float(x) for x in p])


def loadAllForRadiograph(radiographFilename):
    """
    Loads all the landmarks for a given radiograph.
    :return: Dictionary of toothNumber -> Landmark
    """
    radiographFilename = int(radiographFilename)
    landmarks = {}

    for filepath in util.getLandmarkFilenames(radiographFilename):
        filename = os.path.split(filepath)[-1]
        toothNumber = int(re.match("landmarks{}-([0-9]).txt".format(radiographFilename), filename).group(1))
        landmarks[toothNumber] = Landmark(loadLandmarkPoints(filepath), radiographFilename, toothNumber)

    return landmarks
