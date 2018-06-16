import os
import cv2
import glob
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(__file__), "../../", "resources", "data")

def compareSegmentations(numbers):
    numbers = ["%02d" % n for n in numbers] if numbers is not None else []
    truthSegmentationDir = os.path.join(DATA_DIR, "segmentations/")
    ourSegmentationDir = "output/"

    for n in numbers:
        ourPixelsIx = []
        groundTruthPixelsIx =[]
        predicted = glob.glob(ourSegmentationDir+"predicted_{}-*.png".format(n))
        comparison = None
        tmpGroundTruth = None
        tmpPrediction = None
        for i, pred in enumerate(predicted):
            ourImg = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)
            groundTruth = cv2.imread(truthSegmentationDir + "{}-{}.png".format(n, i), cv2.IMREAD_GRAYSCALE)

            if i ==0:
                comparison = np.zeros_like(ourImg)
                tmpGroundTruth = np.zeros_like(ourImg)
                tmpPrediction = np.zeros_like(ourImg)

            tmpGroundTruth += groundTruth
            tmpPrediction += ourImg


        # Our colored pixels
        ourIx = np.where(tmpGroundTruth > 0)
        ourPixelsIx = set(zip(ourIx[0], ourIx[1]))

        # Our black pixels
        ourBlackPixels =  np.where(tmpPrediction == 0)
        ourBlackPixels = set(zip(ourBlackPixels[0], ourBlackPixels[1]))

        # Ground truth colord pixels
        grdTr = np.where(tmpPrediction > 0)
        groundTruthPixelsIx = set(zip(grdTr[0], grdTr[1]))

        # Ground truth black pixels
        groundBlackPixels = np.where(tmpGroundTruth == 0)
        groundBlackPixels = set(zip(groundBlackPixels[0], groundBlackPixels[1]))

        tp = len(ourPixelsIx & groundTruthPixelsIx)
        fp = len(ourPixelsIx - groundTruthPixelsIx)
        tn = len(ourBlackPixels & groundBlackPixels)
        fn = len(ourBlackPixels - groundBlackPixels)

        acc = (tp + tn) / (tp + fp + tn + fn)
        prec = tp / (tp+fp)
        rec = tp / (tp+fn)

        print("Accuracy: {:.2f}%".format(acc * 100))
        print("Precision: {:.2f}%".format(prec * 100))
        print("Recall: {:.2f}%".format(rec * 100))

        print("tp: {}, fp: {}, tn: {}, fn: {}".format(tp, fp, tn, fn))
        finalOverlay = tmpGroundTruth + tmpPrediction
        cv2.imwrite("output/final_{}.png".format(n), finalOverlay)

compareSegmentations(range(1,2))

