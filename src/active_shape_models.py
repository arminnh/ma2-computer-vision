import Radiograph
import util
from ToothModel import ToothModel
from initModel2 import initModel


def buildActiveShapeModels(radiographs, PCAComponents, sampleAmount):
    # 1.1 Load the provided landmarks into your program
    landmarks = []
    for radiograph in radiographs:
        landmarks += list(radiograph.landmarks.values())

    # 1.2 Pre-process the landmarks to normalize translation, rotation, and scale differences
    models = []
    for t in util.TEETH:
        models.append(
            ToothModel(
                name=t,
                landmarks=[l for l in landmarks if l.toothNumber == t],
                pcaComponents=PCAComponents,
                sampleAmount=sampleAmount,
            )
                .buildGrayLevelModels()
                .doProcrustesAnalysis()
        )

    # 1.3 Analyze the data using a Principal Component Analysis (PCA), exposing shape class variations
    for model in models:
        model.doPCA()
        # model.reconstruct()

    # Build gray level model for each point of the mean landmarks of the models
    return models


def buildInitModels(radiographs):
    upper = []
    lower = []
    all = []
    for r in radiographs:
        u = []
        l = []
        a = []
        for i in range(len(r.landmarks)):
            landmark = r.landmarks[i + 1]
            if landmark.toothNumber <= 4:
                u.append(landmark.copy())
            else:
                l.append(landmark.copy())

            a.append(landmark.copy())
        upper.append(u)
        lower.append(l)
        all.append(a)

    # upperModel = initModel(1,upper, range(9,28), util.PCA_COMPONENTS, util.SAMPLE_AMOUNT)
    upperModel = initModel(1, upper, range(0, 40), util.PCA_COMPONENTS, util.SAMPLE_AMOUNT)

    # upperModel.plotLandmarks()
    upperModel = upperModel.buildGrayLevelModels().doProcrustesAnalysis()
    upperModel.doPCA()
    lowerModel = initModel(5, lower, range(0, 40), util.PCA_COMPONENTS, util.SAMPLE_AMOUNT)

    # lowerModel = initModel(5,lower, list(range(0,10)) + list(range(30, 40)), util.PCA_COMPONENTS, util.SAMPLE_AMOUNT)
    # lowerModel.plotLandmarks()
    lowerModel = lowerModel.buildGrayLevelModels().doProcrustesAnalysis()
    lowerModel.doPCA()

    allModel = initModel(-1, all, range(0, 40), util.PCA_COMPONENTS, util.SAMPLE_AMOUNT)
    allModel = allModel.buildGrayLevelModels().doProcrustesAnalysis()
    allModel.doPCA()

    return [upperModel, lowerModel, allModel]


if __name__ == '__main__':
    radiographs = Radiograph.getRadiographs(list(range(4)))

    models = buildActiveShapeModels(radiographs, 20, 2)

    radiographs2 = Radiograph.getRadiographs([14])

    for r in radiographs2:
        model = models[0]
        model.reconstruct(r.landmarks[model.name])
