import os
import sys

from ToothModel import ToothModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import util
from GUI import GUI
import Radiograph
import scripts.goal1
from initModel2 import initModel

def buildInitModels(radiographs):

    upper = []
    lower = []
    all = []
    for r in radiographs:
        u = []
        l = []
        a = []
        for i in range(len(r.landmarks)):
            landmark = r.landmarks[i+1]
            if landmark.toothNumber <= 4:
                u.append(landmark.copy())
            else:
                l.append(landmark.copy())

            a.append(landmark.copy())
        upper.append(u)
        lower.append(l)
        all.append(a)

    #upperModel = initModel(1,upper, range(9,28), util.PCA_COMPONENTS, util.SAMPLE_AMOUNT)
    upperModel = initModel(1,upper, range(0,40), util.PCA_COMPONENTS, util.SAMPLE_AMOUNT)

    #upperModel.plotLandmarks()
    upperModel = upperModel.buildGrayLevelModels().doProcrustesAnalysis()
    upperModel.doPCA()
    lowerModel = initModel(5,lower, range(0,40), util.PCA_COMPONENTS, util.SAMPLE_AMOUNT)

    #lowerModel = initModel(5,lower, list(range(0,10)) + list(range(30, 40)), util.PCA_COMPONENTS, util.SAMPLE_AMOUNT)
    #lowerModel.plotLandmarks()
    lowerModel = lowerModel.buildGrayLevelModels().doProcrustesAnalysis()
    lowerModel.doPCA()

    allModel = initModel(-1 , all, range(0,40), util.PCA_COMPONENTS, util.SAMPLE_AMOUNT)
    allModel = allModel.buildGrayLevelModels().doProcrustesAnalysis()
    allModel.doPCA()

    return [upperModel, lowerModel, allModel]

if __name__ == '__main__':
    radiographNumbers = util.RADIOGRAPH_NUMBERS
    PCAComponents = util.PCA_COMPONENTS
    sampleAmount = util.SAMPLE_AMOUNT

    radiographNumbers = list(range(15))
    radiographs = Radiograph.getRadiographs(radiographNumbers)
    ms = buildInitModels(radiographs)
    models = scripts.goal1.buildActiveShapeModels(radiographs, PCAComponents, sampleAmount)

    # Load other radiographs for GUI but do not load the ones above again
    for radiographNumber in range(15):
        if radiographNumber not in radiographNumbers:
            radiographs.append(Radiograph.getRadiographs([radiographNumber], extra=True)[0])

    gui = GUI(radiographs, models, ms)
    gui.open()

    # print(models)
