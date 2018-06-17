import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import images
import util
import Radiograph
from preprocess_img import *


def main():
    allRadiographs = Radiograph.getRadiographs(util.RADIOGRAPH_NUMBERS)

    r = allRadiographs[0]

    r.showRaw()

    images.preprocessRadiographImage(r.img, [
        bilateralFilter,
        # increaseContrast,
        # tophat,
        applyCLAHE
        # showImg,
    ])

    r.showRaw()


if __name__ == '__main__':
    main()
