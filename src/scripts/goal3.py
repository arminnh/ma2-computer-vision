import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from GUI import GUI
from Radiograph import getRadiographs
import scripts.goal1


def fitModelToImage():
    print("pls")


if __name__ == '__main__':
    models = scripts.goal1.buildActiveShapeModel(list(range(1, 15)))
    r = getRadiographs(list(range(1, 2)))
    gui = GUI(r, models)
    gui.open()

    # print(models)
