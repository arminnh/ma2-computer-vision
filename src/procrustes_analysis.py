import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes

from landmark import Landmark


def listToTuples(p):
    return np.asarray([(float(p[2 * j]), float(p[2 * j + 1])) for j in range(int(len(p) / 2))])


def drawLandmarks(landmarks: List[Landmark], title):
    plt.title(title)
    for l in landmarks:
        points = list(l.getPointsAsTuples())
        points.append(points[0])
        points = np.asarray(points)
        plt.plot(points[:, 0], points[:, 1])
    # plt.plot(x1, y1, x2, y2, marker='o')
    plt.show()


def performProcrustesAnalysis(landmarks: List[Landmark]):
    """
    # https://github.com/prlz77/prlz77.cvtools/blob/master/procrustes_align.py
    :param landmarks: list of landmarks
    """
    # drawLandmarks(landmarks, "procrustes input")

    # First standardize all landmarks
    landmarks = [l.normalize() for l in landmarks]

    # drawLandmarks(landmarks, "normalized")

    # Get a reference
    reference = random.choice(landmarks)
    # scipyProcrustesAnalysis(reference, landmarks)

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

    # drawLandmarks(landmarks, "after Procrustes")

    return landmarks, meanLandmark


def scipyProcrustesAnalysis(reference, landmarks):
    input = landmarks

    d = 1
    while d > 0.0001:
        tempPoints = []
        for l in input:
            m1, m2, diff = procrustes(reference.getPointsAsTuples(), l.getPointsAsTuples())
            tempPoints.append(Landmark(-1, points=m2.flatten()))
            print(diff)
        input = tempPoints

        # Get the new mean
        meanPoints = np.mean(np.asarray([l.getPointsAsList() for l in tempPoints]), axis=0)
        meanLandmark = Landmark(-1, points=meanPoints)

        # Update distance for convergence check
        d = meanLandmark.getShapeDistance(reference)
        reference = meanLandmark

    drawLandmarks(input, "scipy normalized")
