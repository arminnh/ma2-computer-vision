import scipy

from radiograph import Radiograph
from landmark import Landmark
from typing import List
import numpy as np
import random
from scipy.spatial import procrustes
import matplotlib.pyplot as plt

def listToTuples(p):
    return np.asarray([(float(p[2*j]),float(p[2*j+1])) for j in range(int(len(p)/2))])

def drawLandmarks(landmarks: List[Landmark], title):
    plt.title(title)
    for l in landmarks:
        points = list(l.getPointsAsTuples())
        points.append(points[0])
        points = np.asarray(points)
        plt.plot(points[:, 0], points[:, 1])
    #plt.plot(x1, y1, x2, y2, marker='o')
    plt.show()

def performProcrustesAnaylsis(landmarks: List[Landmark]):
    """
    # https://github.com/prlz77/prlz77.cvtools/blob/master/procrustes_align.py
    :param landmarks: list of landmarks
    :return:
    """
    drawLandmarks(landmarks, "procrustes input")

    # First standardize all landmarks
    landmarks = [l.normalize() for l in landmarks]

    drawLandmarks(landmarks, "normalized")

    # Get a reference
    reference = random.choice(landmarks)

    d = 10000
    iteration = 1
    while d > 0.0001:
        print("Iteration: {}".format(iteration))
        # Superimpose all landmarks over the reference
        landmarks = [l.superimpose(reference) for l in landmarks]

        # Get the new mean
        meanPoints = np.mean(np.asarray([l.getPointsAsList() for l in landmarks]), axis=0)
        meanLandmark = Landmark(-1, points=meanPoints)

        # Update distance for convergence check
        d = meanLandmark.getShapeDistance(reference)
        reference = meanLandmark
        iteration += 1

    drawLandmarks(landmarks, "after Procrustes")
    return landmarks
