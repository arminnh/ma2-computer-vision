import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import scripts.goal1


def fitModelToImage():
    print("pls")


if __name__ == '__main__':
    models = scripts.goal1.buildActiveShapeModel()
    print(models)
