import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import images
import Radiograph
from preprocess_img import *


def main():
    r = Radiograph.getRadiographs([1])[0]

    r.showRaw()

    images.preprocessRadiographImage(r.imgPyramid[0], [
        bilateralFilter,
        # increaseContrast,
        # tophat,
        applyCLAHE
    ])

    r.showRaw()


if __name__ == '__main__':
    main()
