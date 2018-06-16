import Radiograph
import util
from ToothModel import ToothModel


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


if __name__ == '__main__':
    radiographs = Radiograph.getRadiographs(list(range(4)))

    models = buildActiveShapeModels(radiographs, 20, 2)

    radiographs2 = Radiograph.getRadiographs([14])

    for r in radiographs2:
        model = models[0]
        model.reconstruct(r.landmarks[model.name])
