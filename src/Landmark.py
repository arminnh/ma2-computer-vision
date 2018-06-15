import math
import os
import re

import numpy as np
import scipy.interpolate

import util


class Landmark:

    def __init__(self, points, radiographFilename=None, toothNumber=None, radiograph=None):
        self.radiographFilename = radiographFilename
        self.toothNumber = toothNumber
        self.points = points if isinstance(points, np.ndarray) else np.array(points)
        self.radiograph = radiograph

    def __str__(self):
        return "Landmark for tooth {} for radiograph {}".format(self.toothNumber, self.radiographFilename)

    def copy(self, points=None):
        points = points if points is not None else self.points
        return Landmark(points, self.radiographFilename, self.toothNumber, self.radiograph)

    def getPointsAsTuples(self):
        p = list(self.points)
        # [ [x_0, y_0], [x_1, y_1] ... ]
        return np.asarray([(float(p[2 * j]), float(p[2 * j + 1])) for j in range(int(len(p) / 2))])

    def getPointsAsList(self):
        return self.points

    def translate(self, x, y):
        """ Translates the points of the landmark. """
        p = self.points.copy()
        p[0::2] = p[0::2] + x
        p[1::2] = p[1::2] + y

        return self.copy(p)

    def scale(self, s):
        """ Scales the points in the landmark by a factor s. """
        return self.copy(self.points * s)

    def rotate(self, theta):
        """ Rotates the points in the landmark. """
        new_points = []
        for p in self.getNormalizedPoints():
            u = math.cos(theta) * p[0] - math.sin(theta) * p[1]
            v = math.sin(theta) * p[0] + math.cos(theta) * p[1]

            new_points.append(u)
            new_points.append(v)

        return self.copy(np.asarray(new_points))

    def getMeanShiftedPoints(self):
        """ Returns the landmark points translated by their means. """
        # https://en.wikipedia.org/wiki/Procrustes_analysis
        p = self.getPointsAsTuples()
        translateXY = -np.mean(p, axis=0)
        return p + translateXY, translateXY

    def getScale(self):
        """ Returns a statistical measure of the object's scale, root mean square distance (RMSD). """
        # https://en.wikipedia.org/wiki/Procrustes_analysis
        # TODO: check if this scaling factor is correct, maybe scale using SVD
        distance, _ = self.getMeanShiftedPoints()
        return np.sqrt(np.mean(np.square(distance)))

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
        """ Returns the SSD from an other landmark. """
        return np.sqrt(np.sum(np.square(self.getPointsAsList() - other.getPointsAsList())))

    def getNormalizedPoints(self):
        """ Returns an array of normalized landmark points.  """
        meanShiftedPoints, _ = self.getMeanShiftedPoints()
        return meanShiftedPoints / self.getScale()

    def normalize(self):
        """ Normalizes the points in the landmark. """
        return self.copy(self.getNormalizedPoints().flatten())

    def superimpose(self, other):
        """ Returns this landmark superimposed (translated, scaled, and rotated) over another. """
        meanShiftedPoints, translationXY = self.getMeanShiftedPoints()
        theta = self.getThetaForReference(other)
        s = self.getScale()

        superimposed = self.translate(*translationXY).scale(1/s).rotate(theta)

        return superimposed, translationXY, s, theta

    def normalSamplesForAllPoints(self, pixelsToSample):
        """
        Returns samples from the normal lines on the landmark points
        """
        lines = {}

        points = self.getPointsAsTuples()
        for i, point in enumerate(points):
            m = util.getNormalSlope(points[i - 1], point, points[(i + 1) % len(points)])
            normalPoints = util.sampleLine(m, point, pixelsToSample=pixelsToSample)

            lines[i] = normalPoints

        return lines

    def getGrayLevelProfilesForAllNormalPoints(self, sampleAmount, getDeriv=True):
        if self.radiograph is None:
            raise Exception("Need radiograph for gray level profile")

        normalizedGrayLevelProfilesWithPoints = {}

        points = self.getPointsAsTuples()
        for i, point in enumerate(points):
            # Build gray level profile by sampling a few points on each side of the point.

            # Sample points on normal line of the current landmark point
            m = util.getNormalSlope(points[i - 1], point, points[(i + 1) % len(points)])
            #tangentLineSlope = f(point[0])
            # m = slope of normal line
            #m = -1 / tangentLineSlope if tangentLineSlope != 0 else 0

            normalSamplePoints = util.sampleLine(m, point, pixelsToSample=sampleAmount)
            normalizedGrayLevelProfilesWithPoints[i] = []

            # Loop over the sampled points
            # We need to get the gray level profile of all these points
            for normalPoint in normalSamplePoints:
                # Get pixel values on the sampled positions
                p2 = util.sampleLine(m, normalPoint, pixelsToSample=sampleAmount)

                beforeDeriv, afterDeriv, scaled = util.getPixels(self.radiograph, p2, getDeriv)

                #img = self.radiograph.image#.convert("L")  # type: Image
                #pixels = np.asarray([img.getpixel(p) for p in p2])

                if getDeriv:
                    pixels = scaled
                    # Derivative profile of length n_p - 1
                    #pixels = np.asarray([pixels[i+1] - pixels[i-1] for i in range(len(pixels)-1)])#np.diff(pixels)

                    # Normalized derivative profile
                    #scale = np.sum(np.abs(pixels))
                    #if scale != 0:
                    #    pixels = pixels / scale
                else:
                    pixels = beforeDeriv

                normalizedGrayLevelProfilesWithPoints[i].append([pixels, normalPoint, p2])
            # print("PROFILES SHAPE: ", pixels.shape)

        return normalizedGrayLevelProfilesWithPoints

    def grayLevelProfileForAllPoints(self, pixelsToSample, getDeriv=True):
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
        normalizedGrayLevelProfiles = {}
        normalPointsOfLandmarkNr = {}
        points = self.getPointsAsTuples()

        for i, point in enumerate(points):
            # Build gray level profile by sampling a few points on each side of the point.

            # Sample points on normal line of the current landmark point
            m = util.getNormalSlope(points[i - 1], point, points[(i + 1) % len(points)])
            #tangentLineSlope = f(point[0])
            # m = slope of normal line
            #m = -1 / tangentLineSlope if tangentLineSlope != 0 else 0

            normalPoints = util.sampleLine(m, point, pixelsToSample=pixelsToSample)

            _, afterDeriv, scaled = util.getPixels(self.radiograph, normalPoints, getDeriv)
            grayLevelProfiles[i] = afterDeriv
            # Get pixel values on the sampled positions
            #img = self.radiograph.image  # type: Image
            #pixels = np.asarray([img.getpixel(p) for p in normalPoints])
            #
            # if getDeriv:
            #     # Derivative profile of length n_p - 1
            #     pixels = np.asarray([pixels[i+1] - pixels[i-1] for i in range(len(pixels)-1)])#np.diff(pixels)
            #
            # grayLevelProfiles[i] = pixels
            #
            # # Normalized derivative profile
            # # print("i {}, derivated profile: {}, divisor: {}".format(i, list(pixels), np.sum(np.abs(pixels))), end=", ")
            # scale = np.sum(np.abs(pixels))
            # if scale != 0:
            #     pixels = pixels / scale
            # # print("normalized profile: {}".format(list(pixels)))

            normalizedGrayLevelProfiles[i] = scaled
            normalPointsOfLandmarkNr[i] = normalPoints
            # print("PROFILES SHAPE: ", pixels.shape)

        return grayLevelProfiles, normalizedGrayLevelProfiles, normalPointsOfLandmarkNr

    def calculateDerivative(self, points):
        xx = points[:, 0]
        yy = points[:, 1]
        sorted_xx = xx.argsort()
        # Fuck you scipy and your strictly increasing x values
        xx = xx[sorted_xx]
        for i in range(len(xx)):
            xx[i] += 0.001 * (i + 1)
        yy = yy[sorted_xx]
        # y = m x + b

        f = scipy.interpolate.CubicSpline(xx, yy).derivative()
        return f

def loadLandmarkPoints(filename):
    f = open(filename, "r")
    p = f.readlines()
    return np.asarray([float(x) for x in p])


def loadAllForRadiograph(radiographFilename, XOffset, YOffset):
    """
    Loads all the landmarks for a given radiograph.
    :return: Dictionary of toothNumber -> Landmark
    """
    radiographFilename = int(radiographFilename)
    landmarks = {}

    for filepath in util.getLandmarkFilenames(radiographFilename):
        filename = os.path.split(filepath)[-1]
        toothNumber = int(re.match("landmarks{}-([0-9]).txt".format(radiographFilename), filename).group(1))
        landmarks[toothNumber] = Landmark(loadLandmarkPoints(filepath), radiographFilename, toothNumber).translate(XOffset, YOffset)

    return landmarks
