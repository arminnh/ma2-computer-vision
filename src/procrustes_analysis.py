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
    # First standardize all landmarks
    for l in landmarks:
        l.normalize()

    drawLandmarks(landmarks, "pre")

    # Get a reference
    ref = random.choice(landmarks).getPointsAsList()

    d = 10000
    iteration = 1
    while d > 0.0001:
        print("Iteration: {}".format(iteration))
        mean = []
        for l in landmarks:
            # Moet dit hier?
            # Volgens "Active shape modelling - their training and applications" wel.
            l.normalize()

            theta = l.getThetaForReference(listToTuples(ref))
            l.rotate(theta)
            mean.append(l.getPointsAsList())

        # Get the new mean
        new_mean = np.mean(np.asarray(mean), axis=0)
        # Check for convergence
        # Convergence is defined as not changing that much between old mean and curr mean
        d = getDistance(ref, new_mean)

        ref = new_mean
        iteration += 1

    drawLandmarks(landmarks, "post")
    return landmarks

def getDistance(ref, mean):
    return np.sqrt(np.sum(np.square(ref - mean)))