import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import Radiograph
from preprocess_img import *


def runExercise():
    allRadiographs = Radiograph.getRadiographs()

    r = allRadiographs[0]
    r.showRaw()
    r.preprocessRadiograph([
        bilateralFilter,
        # increaseContrast,
        # tophat,
        applyCLAHE
        # showImg,
    ])
    r.showRaw()
