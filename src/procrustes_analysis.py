import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes

from Landmark import Landmark


def listToTuples(p):
    return np.asarray([(float(p[2 * j]), float(p[2 * j + 1])) for j in range(int(len(p) / 2))])


def plotLandmarks(landmarks: List[Landmark], title):
    plt.figure()
    plt.title(title)
    ax = plt.gca()

    for l in landmarks:
        points = l.getPointsAsTuples()
        plt.plot(points[:, 0], points[:, 1])

    ax.set_ylim(ax.get_ylim()[::-1])
    plt.axis("equal")
    plt.show()


def performProcrustesAnalysis(landmarks: List[Landmark]):
    # drawLandmarks(landmarks, "procrustes input")

    # Get a reference
    reference = random.choice(landmarks)
    meanLandmark = reference
    # scipyProcrustesAnalysis(reference, landmarks)

    mean_theta = 0
    mean_scale = reference.getScale()
    reference = reference.normalize()

    d = 10000
    iteration = 1
    while d > 0.0001:
        # Superimpose all landmarks over the reference
        newLandmarks = []
        for landmark in landmarks:
            landmark, translationXY, scale, theta = landmark.superimpose(reference)
            newLandmarks.append(landmark)
            mean_theta += theta
            mean_scale += scale

        landmarks = newLandmarks

        # Get the new mean
        meanPoints = np.mean(np.asarray([l.points for l in landmarks]), axis=0)
        meanLandmark = Landmark(points=meanPoints)

        # Update distance for convergence check
        d = meanLandmark.shapeDistance(reference)
        reference = meanLandmark
        iteration += 1

    # drawLandmarks(landmarks, "after Procrustes")

    mean_theta /= len(landmarks)
    mean_scale /= len(landmarks)
    return landmarks, meanLandmark, mean_scale, mean_theta


def scipyProcrustesAnalysis(reference, landmarks):
    input = landmarks

    d = 1
    while d > 0.0001:
        tempPoints = []
        for l in input:
            m1, m2, diff = procrustes(reference.getPointsAsTuples(), l.getPointsAsTuples())
            tempPoints.append(Landmark(m2.flatten()))
            print(diff)
        input = tempPoints

        # Get the new mean
        meanPoints = np.mean(np.asarray([l.points for l in tempPoints]), axis=0)
        meanLandmark = Landmark(meanPoints)

        # Update distance for convergence check
        d = meanLandmark.shapeDistance(reference)
        reference = meanLandmark

    plotLandmarks(input, "scipy normalized")
