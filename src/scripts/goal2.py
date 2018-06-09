import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from helpers import *
from preprocess_img import *


def runExercise():
    allRadiographs = getAllRadiographs()

    r = allRadiographs[0]
    r.showRawRadiograph()
    r.preprocessRadiograph([
        PILtoCV,
        bilateralFilter,
        # increaseContrast,
        # tophat,
        applyCLAHE,
        # showImg,
        cvToPIL
    ])
    r.showRawRadiograph()
