import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.decomposition import PCA
from helpers import *
from radiograph import *
import procrustes_analysis


def runExercise():
    allRadiographs = getAllRadiographs()
    # 1.1 load all provided landmarks
    allLandmarks = getAllLandmarks(allRadiographs)

    # 1.2 pre-process landmarks
    models = {}
    for k1, v1 in TOOTH_TYPES2.items():
        for k2, v2 in TOOTH_TYPES1.items():
            modelName = k1 + "-" + k2
            if modelName == "UPPER-LATERAL":
                models[modelName] = [l for l in allLandmarks if l.toothNumber in v1 & v2]

    for model, landmarks in models.items():
        newLandmarks, meanLandmark = procrustes_analysis.performProcrustesAnalysis(landmarks)
        models[model] = (newLandmarks, meanLandmark)

    # 1.3 perform PCA
    for model, (newLandmarks, meanLandmark) in models.items():
        data = [l.getPointsAsList() for l in newLandmarks[:-1]]
        components = 20
        p = PCA(n_components=components)

        p.fit(data)
        P = p.components_

        x = newLandmarks[-1]
        b = np.matmul(P, x.points - meanLandmark.points).reshape((components, -1))

        print("b: ", b.shape)
        print("P: ", P.shape)
        newLandmark = meanLandmark.points + np.matmul(np.transpose(P), b).flatten()

        procrustes_analysis.drawLandmarks([x], "INPUT")
        procrustes_analysis.drawLandmarks([Landmark(-1, points=newLandmark)], "PCA generated")


if __name__ == '__main__':
    runExercise()
