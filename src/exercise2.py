from helpers import *
from preprocess_img import *
def runExercise():
    allRadiographs = getAllRadiographs()


    r = allRadiographs[0]
    r.showRawRadiograph()
    r._preprocessRadiograph([PILtoCV,
                             bilateralFilter,
                             #increaseContrast,
                             #tophat,
                             applyCLAHE,
                             #showImg,
                             cvToPIL])
    r.showRawRadiograph()
