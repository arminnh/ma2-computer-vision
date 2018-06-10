import os
import sys
from Radiograph import Radiograph, getRadiographs
from GUI import GUI
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import scripts.goal1


def fitModelToImage():
    print("pls")


if __name__ == '__main__':
    models = scripts.goal1.buildActiveShapeModel()

    r = getRadiographs(list(range(1, 3)))
    gui = GUI(r, models)
    gui.open()



    #print(models)
