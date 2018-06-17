import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import glob

import cv2
import numpy as np

import Radiograph
import util
from models import MultiResolutionASM

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../", "resources", "data")


class MaskGenerator:

    def __init__(self, radiograph, multiResolutionModel):
        self.meanSplitLine = int(np.mean(radiograph.jawSplitLine[:, 1]))
        self.currentRadiograph = radiograph
        self.img = radiograph.imgPyramid[0]
        self.multiResolutionModel = multiResolutionModel
        self.currentResolutionLevel = self.multiResolutionModel.resolutionLevels - 1
        self.currentLandmark = None

    def setCurrentResolutionLevel(self, level):
        if level >= self.multiResolutionModel.resolutionLevels:
            level = self.multiResolutionModel.resolutionLevels - 1

        if level < 0:
            level = 0

        if self.currentLandmark is not None:
            if level > self.currentResolutionLevel:
                self.currentLandmark = self.currentLandmark.scale(0.5)
            elif level < self.currentResolutionLevel:
                self.currentLandmark = self.currentLandmark.scale(2)

        self.currentResolutionLevel = level

        return self

    def initializeLandmark(self):
        _, xMax = self.currentRadiograph.imgPyramid[self.currentResolutionLevel].shape

        meanSplitLine = self.currentRadiograph.jawSplitLine[:, 1].mean() * 0.5 ** self.currentResolutionLevel
        mouthMiddle = int(round(meanSplitLine))

        # print("Initialization at ({}, {})".format((xMax / 2), mouthMiddle))

        self.currentLandmark = self.multiResolutionModel.getTranslatedAndInverseScaledMeanMouth(
            self.currentResolutionLevel, (xMax / 2), mouthMiddle
        )

    def multiResolutionSearch(self):
        self.setCurrentResolutionLevel(self.multiResolutionModel.resolutionLevels)
        self.initializeLandmark()

        for level in range(self.multiResolutionModel.resolutionLevels - 1, -1, -1):
            print("Multi resolution search: resolution {}".format(level))
            self.setCurrentResolutionLevel(level)

            self.currentLandmark = self.multiResolutionModel.improveLandmarkForResolutionLevel(
                resolutionLevel=level,
                img=self.currentRadiograph.imgPyramid[level].copy(),
                landmark=self.currentLandmark
            )

    def createMasksForTeeth(self):
        # get offsets for image
        (offsetX, offsetY) = self.currentRadiograph.offsets

        # Do complete multi resolution search
        self.multiResolutionSearch()

        # Create empty mask
        mask = np.zeros_like(self.currentRadiograph.origImg)

        # get landmark for each tooth
        points = self.currentLandmark.getPointsAsTuples().round().astype(int)

        for i in range(int(len(points) / 40)):
            toothPoints = points[i * 40:(i + 1) * 40]
            toothPoints = np.asarray([(-offsetX + int(p[0]), -offsetY + int(p[1])) for p in toothPoints])

            cv2.fillPoly(mask, [toothPoints], 255)

        masked_image = cv2.bitwise_and(self.currentRadiograph.origImg, mask)
        masked_image[np.where(masked_image > 0)] = 255

        return masked_image

    def doSearchAndCompareSegmentations(self):
        predictions = self.createMasksForTeeth()

        groundTruth = np.zeros_like(self.currentRadiograph.origImg)
        truthSegmentationDir = os.path.join(DATA_DIR, "segmentations/")
        originalSegments = glob.glob(truthSegmentationDir + "{}-*.png".format(self.currentRadiograph.number))
        for original in originalSegments:
            toothGroundTruth = cv2.imread(original, cv2.IMREAD_GRAYSCALE)

            groundTruth += toothGroundTruth

        finalOverlay = groundTruth + predictions
        cv2.imwrite("output/final_{}_multi_resolution.png".format(self.currentRadiograph.number), finalOverlay)

        # Our colored pixels
        ourIx = np.where(predictions > 0)
        ourPixelsIx = set(zip(ourIx[0], ourIx[1]))

        # Our black pixels
        ourBlackPixels = np.where(predictions == 0)
        ourBlackPixelsIx = set(zip(ourBlackPixels[0], ourBlackPixels[1]))

        # Ground truth colored pixels
        grdTr = np.where(groundTruth > 0)
        groundTruthPixelsIx = set(zip(grdTr[0], grdTr[1]))

        # Ground truth black pixels
        groundBlackPixels = np.where(groundTruth == 0)
        groundBlackPixelsIx = set(zip(groundBlackPixels[0], groundBlackPixels[1]))

        tp = len(ourPixelsIx & groundTruthPixelsIx)
        fp = len(ourPixelsIx - groundTruthPixelsIx)
        tn = len(ourBlackPixelsIx & groundBlackPixelsIx)
        fn = len(ourBlackPixelsIx - groundBlackPixelsIx)
        print("tp: {}, fp: {}, tn: {}, fn: {}".format(tp, fp, tn, fn))

        acc = (tp + tn) / (tp + fp + tn + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        print("Accuracy = {:.2f}%, Precision = {:.2f}%, Recall = {:.2f}%".format(acc * 100, prec * 100, rec * 100))

        return acc, prec, rec


def leaveOneOutCrossValidation(radiographs, resolutionLevels, maxLevelIterations, grayLevelModelSize, sampleAmount,
                               PCAComponents):
    print("\nSTARTING LEAVE ONE OUT CROSS VALIDATION.\n")
    accuracy = 0
    precision = 0
    recall = 0

    mirroredRadiographs = [r for r in radiographs if r.mirrored]
    radiographs = [r for r in radiographs if not r.mirrored]

    for i in range(len(radiographs)):
        trainSet = mirroredRadiographs + radiographs[:i] + radiographs[i + 1:]
        testSet = [radiographs[i]]

        with util.Timer("Building multi resolution active shape model, leaving radiograph {} out".format(i)):
            model = MultiResolutionASM.buildModel(trainSet, resolutionLevels, maxLevelIterations, grayLevelModelSize,
                                                  sampleAmount, PCAComponents)

        with util.Timer("Generating segmentation masks and comparing to ground truth".format(i)):
            gen = MaskGenerator(testSet[0], model)
            acc, prec, rec = gen.doSearchAndCompareSegmentations()

        print()
        accuracy += acc
        precision += prec
        recall += rec

    accuracy /= len(radiographs)
    precision /= len(radiographs)
    recall /= len(radiographs)
    print("Averaged results after LOO cross-validation:")
    print("Accuracy = {:.2f}%, Precision = {:.2f}%, Recall = {:.2f}%"
          .format(accuracy * 100, precision * 100, recall * 100))


if __name__ == '__main__':
    radiographNumbers = list(range(20))
    PCAComponents = 25
    sampleAmount = 3
    maxLevelIterations = 20
    grayLevelModelSize = 7
    resolutionLevels = 5

    with util.Timer("Loading images"):
        radiographs = Radiograph.getRadiographs(
            numbers=radiographNumbers,
            extra=False,
            resolutionLevels=resolutionLevels,
            withMirrored=True
        )

    leaveOneOutCrossValidation(radiographs, resolutionLevels, maxLevelIterations, grayLevelModelSize, sampleAmount,
                               PCAComponents)
