import math
import os
import re

import numpy as np

import images
import util


class Landmark:

    def __init__(self, points, radiographFilename=None, toothNumber=None):
        self.radiographFilename = radiographFilename
        self.radiograph = None
        self.toothNumber = toothNumber
        self.points = points if isinstance(points, np.ndarray) else np.array(points)

    def __str__(self):
        return "Landmark for tooth {} for radiograph {}".format(self.toothNumber, self.radiographFilename)

    def copy(self, points=None):
        points = points if points is not None else self.points
        return Landmark(points, self.radiographFilename, self.toothNumber)

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
        p = self.getPointsAsTuples()
        translateXY = -np.mean(p, axis=0)
        return p + translateXY, translateXY

    def getScale(self):
        """ Returns a statistical measure of the object's scale, root mean square distance (RMSD). """
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

        superimposed = self.translate(*translationXY).scale(1 / s).rotate(theta)

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

    def getGrayLevelProfilesForNormalPoints(self, img, sampleAmount, derive):
        profilesForLandmarkPoints = {}

        points = self.getPointsAsTuples()
        for i, point in enumerate(points):
            # Each landmark point will have a list of gray level models
            # One gray level model for points on the normal line on the current landmark point
            profilesForLandmarkPoints[i] = []

            # Sample points on normal line (slope m) of the current landmark point
            m = util.getNormalSlope(points[i - 1], point, points[(i + 1) % len(points)])
            normalPoints = util.sampleLine(m, point, pixelsToSample=sampleAmount)
            print("Normal points for landmark point {}: {}".format(i, normalPoints))

            # Loop over the sampled points and get the gray level profile of all them
            for normalPoint in normalPoints:
                # Get pixel values on the sampled positions
                grayLevelProfilePoints = util.sampleLine(m, normalPoint, pixelsToSample=sampleAmount)

                rawPixelProfile, derivedProfile, normalizedProfile \
                    = images.getPixelProfile(img, grayLevelProfilePoints, derive)

                if derive:
                    grayLevelProfile = normalizedProfile
                else:
                    grayLevelProfile = rawPixelProfile

                profilesForLandmarkPoints[i].append([grayLevelProfile, normalPoint, grayLevelProfilePoints])

        return profilesForLandmarkPoints

    def grayLevelProfileForLandmarkPoints(self, img, sampleAmount):
        """
        For every landmark point j (all points in this landmark) in the image i (the radiograph of this landmark) of
        the training set, we extract a gray level profile g_ij of length n_p pixels, centered around the landmark point.
        Not the actual gray level profile but its normalized derivative to get invariance to the offsets and uniform
        scaling of the gray level.
        The gray level profile of a landmark point j is a vector of n_p values.
        ~ "Active Shape Models - Part I: Modeling Shape and Gray Level Variations"
        """
        derivedGrayLevelProfiles = {}
        normalizedGrayLevelProfiles = {}
        normalPointsForLandmarkPoints = {}
        points = self.getPointsAsTuples()

        for i, point in enumerate(points):
            # Build gray level profile by sampling a few points on each side of the point.

            # Sample points on normal line of the current landmark point
            m = util.getNormalSlope(points[i - 1], point, points[(i + 1) % len(points)])
            normalPoints = util.sampleLine(m, point, pixelsToSample=sampleAmount)

            _, derivedProfile, normalizedProfile = images.getPixelProfile(img, normalPoints, derive=True)

            derivedGrayLevelProfiles[i] = derivedProfile
            normalizedGrayLevelProfiles[i] = normalizedProfile
            normalPointsForLandmarkPoints[i] = normalPoints

        return derivedGrayLevelProfiles, normalizedGrayLevelProfiles, normalPointsForLandmarkPoints


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
        landmarks[toothNumber] = Landmark(loadLandmarkPoints(filepath), radiographFilename, toothNumber)\
            .translate(XOffset, YOffset)

    return landmarks
