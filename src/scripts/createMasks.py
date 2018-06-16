import cv2
import numpy as np

import Radiograph
import util
import models.active_shape_model as asm

class MaskGenerator:

    def __init__(self, radiograph, models, incisorModels):
        self.meanSplitLine = int(np.mean(radiograph.jawSplitLine[:,1]))
        self.currentRadiograph = radiograph
        self.currentToothModel = None
        self.incisorModels = incisorModels
        self.img = radiograph.img
        self.toothModels = models
        self.currentInitLandmark = None
        self.currentLandmark = None

    def initIncisorModels(self, modelNr=0):
        _,x = self.img.shape
        self.currentToothModel = self.incisorModels[modelNr]

        self.currentLandmark = self.incisorModels[modelNr].initLandmark(self.meanSplitLine, x)
        #centers = self.incisorModels[model].getCentersOfInitModel(self.currentLandmark)

        #for c in centers:
        #    orig = (int(c[0]), int(c[1]))
        #    cv2.circle(self.img, orig, 20, 255, 3)

    def autoFitToothModel(self):
        currentToothModel = self.currentToothModel

        optim = False
        # Check if we can optimize
        if self.currentInitLandmark:
            if (self.currentInitLandmark.toothNumber == 1 and currentToothModel.name <= 4) or \
                    (self.currentInitLandmark.toothNumber == 5 and currentToothModel.name > 4):
                # Ok optimize!
                optim = True

        if not optim:
            # First set correct initialisation model
            if currentToothModel.name <= 4:
                # Upper jaw
                self.initIncisorModels(0)
            else:
                # lower jaw
                self.initIncisorModels(1)

            # Now we fit it
            self.findBetterLandmark()

            self.currentInitLandmark = self.currentLandmark

            # Once this is done, we get the origins
            origins = util.getCentersOfInitModel(self.currentLandmark)

        else:
            # Optimize by just getting the centers of the saved init model
            origins = util.getCentersOfInitModel(self.currentInitLandmark)

        # Now we reset the correct tooth model
        self.currentToothModel = currentToothModel

        # Get the starting pos of the tooth
        (x, y) = origins[(currentToothModel.name - 1) % len(origins)]

        # init tooth at that position
        self.currentLandmark = self.currentToothModel.getTranslatedAndInverseScaledMean(x, y)

        # Iterate with the tooth
        self.findBetterLandmark()

    def findBetterLandmark(self):
        """
        Execute an iteration of "matching model points to target points"
        model points are defined by the model, target points by the 'bestLandmark'
        """
        previousLandmark = self.currentLandmark
        d = 2
        i = 0

        if self.currentToothModel.name <= 4:
            jawImg = self.currentRadiograph.imgUpperJaw
        else:
            jawImg = self.currentRadiograph.imgLowerJaw

        if self.currentToothModel.name == -1:
            jawImg = self.currentRadiograph.img

        while i < 10:
            newTargetPoints = self.currentToothModel.findBetterFittingLandmark(jawImg, previousLandmark)
            improvedLandmark = self.currentToothModel.matchModelPointsToTargetPoints(newTargetPoints)

            d = improvedLandmark.shapeDistance(previousLandmark)
            previousLandmark = improvedLandmark

            i += 1
            print("Improvement iteration {}, distance {}".format(i, d))

        self.currentLandmark = previousLandmark

    def generateMask(self):
        # get offsets
        (offsetX, offsetY) = self.currentRadiograph.offsets

        # Save current model
        oldModel = self.currentToothModel

        # For each model we will get a landmark
        for i, model in enumerate(self.toothModels):
            # Create empty mask
            mask = np.zeros_like(self.currentRadiograph.origImg)

            # Set currentToothModel
            self.currentToothModel = model

            # Create a landmark for this model
            self.autoFitToothModel()

            # Get the current landmark
            landmark = self.currentLandmark

            points = np.asarray([(-offsetX + int(p[0]), -offsetY + int(p[1])) for p in landmark.getPointsAsTuples()])

            cv2.fillPoly(mask, [points], 255)

            masked_image = cv2.bitwise_and(self.currentRadiograph.origImg, mask)
            masked_image[np.where(masked_image > 0)] = 255

            cv2.imwrite('output/predicted_{}-{}.png'.format(self.currentRadiograph.number, i), masked_image)

        self.currentToothModel = oldModel



if __name__ == '__main__':
    radiographNumbers = util.RADIOGRAPH_NUMBERS
    PCAComponents = util.PCA_COMPONENTS
    sampleAmount = util.SAMPLE_AMOUNT

    with util.Timer("Loading images"):
        radiographNumbers = list(range(15))
        radiographs = Radiograph.getRadiographs(radiographNumbers)

    initModels = asm.buildInitModels(radiographs)
    with util.Timer("Building active shape models"):
        models = asm.buildActiveShapeModels(radiographs, PCAComponents, sampleAmount)

    # Load other radiographs for GUI but do not load the ones above again
    with util.Timer("Loading remaining images (without landmarks)"):
        for radiographNumber in range(30):
            if radiographNumber not in radiographNumbers:
                radiographs.append(Radiograph.getRadiographs([radiographNumber], extra=True)[0])


    for r in radiographs:
        gen = MaskGenerator(r, models, initModels)
        gen.generateMask()

    # print(models)
