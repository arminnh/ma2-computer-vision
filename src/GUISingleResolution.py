import images
import util
from preprocess_img import *


class GUISingleResolution:

    def __init__(self, radiographs, models, incisorModels):
        self.name = "Computer Vision KU Leuven"
        self.radiographs = radiographs
        self.currentRadiograph = None
        self.currentRadiographIndex = 0
        self.toothModels = models
        self.currentToothModel = None
        self.currentToothModelIndex = 0
        self.toothCenters = self.getAllToothCenters()
        self.incisorModels = incisorModels
        self.img = None
        self.currentInitLandmark = None
        self.preprocess = False
        self.currentLandmark = None
        self.showEdges = False

    def open(self):
        self.createWindow()
        self.setCurrentImage(self.currentRadiographIndex)
        self.setCurrentModel(self.currentToothModelIndex)
        self.initTrackBars()

        while True:
            # clone the img
            img = self.img.copy()

            if self.showEdges:
                img = self.drawEdges(img)

            y, x = img.shape
            cv2.line(img, (int(x / 2), 0), (int(x / 2), int(y)), 255, 2)
            cv2.line(img, (0, int(y / 2)), (x, int(y / 2)), 255, 2)

            # for model in self.toothModels:
            #     if self.currentRadiographIndex < len(model.landmarks):
            #         landmark = model.landmarks[self.currentRadiographIndex]
            #         self.drawLandmark(landmark, color=180, thickness=3)
            #
            #         points, profile = model.initializationModel.grayLevelProfileForImage[self.currentRadiographIndex]
            #         self.drawGreyLevelProfile(points, profile)
            #         self.drawToothCenters()

            cv2.imshow(self.name, img)

            # Key Listeners
            pressed_key = cv2.waitKey(50)

            # quit
            if pressed_key == ord("q"):
                break

            # select another radiograph
            elif pressed_key == ord("x") or pressed_key == ord("c"):
                if len(self.radiographs) > 1:
                    if pressed_key == ord("x"):
                        self.increaseRadiographIndex(-1)
                    elif pressed_key == ord("c"):
                        self.increaseRadiographIndex(1)
                    cv2.setTrackbarPos("radiograph", self.name, self.currentRadiographIndex)

            # select another model
            if pressed_key == ord("v") or pressed_key == ord("b"):
                if len(self.radiographs) > 1:
                    if pressed_key == ord("v"):
                        self.increaseModelIndex(-1)
                    elif pressed_key == ord("b"):
                        self.increaseModelIndex(1)
                    cv2.setTrackbarPos("model", self.name, self.currentToothModelIndex)

            # show edges key listener
            elif pressed_key == ord("e"):
                if self.showEdges:
                    self.showEdges = False
                    cv2.displayOverlay(self.name, "Edges turned OFF!", 1000)

                else:
                    self.showEdges = True
                    cv2.displayOverlay(self.name, "Edges turned ON!", 1000)

            elif pressed_key == ord("p"):
                self.preprocessCurrentRadiograph()

            elif pressed_key == ord("o"):
                self.findBetterToothCenters()

            elif pressed_key == ord("n"):
                if self.currentLandmark is None:
                    cv2.displayOverlay(self.name, "No landmark set yet!", 1000)
                else:
                    self.findBetterLandmark()

            elif pressed_key == ord("u"):
                self.showUpperJaw()

            elif pressed_key == ord("i"):
                self.initIncisorModels()

            elif pressed_key == ord("l"):
                self.showLowerJaw()

            elif pressed_key == ord("s"):
                self.setIncisorModel()

            elif pressed_key == ord("a"):
                self.autoFitToothModel()

            elif pressed_key == ord("m"):
                self.getMaskForRadiograph()

            if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

    def createWindow(self):
        cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.name, 1000, 700)
        cv2.setMouseCallback(self.name, self.mouseListener)
        return self

    def refreshOverlay(self):
        modelName = self.currentToothModel.name if self.currentToothModel is not None else self.currentToothModelIndex

        cv2.displayOverlay(
            self.name,
            "Showing image {}, model {}".format(self.currentRadiographIndex, modelName),
            1000
        )
        return self

    def refreshCurrentImage(self):
        self.img = self.currentRadiograph.imgPyramid[0].copy()

        jawSplitLine = self.currentRadiograph.jawSplitLine
        for i, (x, y) in enumerate(jawSplitLine):
            if i > 0:
                cv2.line(self.img, (jawSplitLine[i - 1][0], jawSplitLine[i - 1][1]), (x, y), 255, 2)
        self.meanSplitLine = int(np.mean(jawSplitLine[:, 1]))
        cv2.line(self.img, (0, self.meanSplitLine), (self.img.shape[1], self.meanSplitLine), 200, 2)

    def setCurrentImage(self, idx):
        self.currentRadiographIndex = idx
        self.currentRadiograph = self.radiographs[idx]
        self.toothCenters = self.getAllToothCenters()
        self.refreshCurrentImage()
        self.refreshOverlay()

        return self

    def setCurrentModel(self, idx):
        self.currentToothModelIndex = idx
        self.currentToothModel = self.toothModels[idx]
        self.refreshOverlay()
        return self

    def initTrackBars(self):
        if len(self.radiographs) > 1:
            cv2.createTrackbar("radiograph", self.name, self.currentRadiographIndex, len(self.radiographs) - 1,
                               self.setCurrentImage)

        if len(self.toothModels) > 1:
            cv2.createTrackbar("model", self.name, self.currentToothModelIndex, len(self.toothModels) - 1,
                               self.setCurrentModel)

        return self

    def drawEdges(self, tmp_img):
        blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
        edges = cv2.Canny(blur, 20, 60)
        # Overlap image and edges together
        tmp_img = np.bitwise_or(tmp_img, edges)
        # tmp_img = cv2.addWeighted(tmp_img, 1 - edges_val, edges, edges_val, 0)
        return tmp_img

    def increaseRadiographIndex(self, amount):
        self.preprocess = False
        self.currentRadiographIndex += amount

        if self.currentRadiographIndex < 0:
            self.currentRadiographIndex = 0

        if self.currentRadiographIndex >= len(self.radiographs):
            self.currentRadiographIndex = len(self.radiographs) - 1

        self.currentInitLandmark = None
        return self

    def increaseModelIndex(self, amount):
        self.currentToothModelIndex += amount

        if self.currentToothModelIndex < 0:
            self.currentToothModelIndex = 0

        if self.currentToothModelIndex > len(self.toothModels):
            self.currentToothModelIndex = len(self.toothModels) - 1

        return self

    # mouse callback function
    def mouseListener(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.currentLandmark = self.currentToothModel.getTranslatedAndInverseScaledMean(x, y)
            self.drawLandMarkWithNormals(self.currentLandmark, grayLevels=True)

    def drawLandmark(self, landmark, color, thickness=1):
        points = landmark.getPointsAsTuples().round().astype(int)

        for i in range(len(points)):
            start = (points[i][0], points[i][1])
            end = (points[(i + 1) % len(points)][0], points[(i + 1) % len(points)][1])

            cv2.line(self.img, start, end, color, thickness)

    def drawLandMarkWithNormals(self, landmark, grayLevels, color=255):
        points = landmark.getPointsAsTuples().round().astype(int)

        for i in range(len(points)):
            m = util.getNormalSlope(points[i - 1], points[i], points[(i + 1) % len(points)])
            p = np.asarray(util.sampleLine(m, points[i], self.currentToothModel.sampleAmount))
            x2 = p[:, 0]
            y2 = p[:, 1]

            for j in range(len(x2)):
                cv2.line(self.img, (x2[j], y2[j]), (x2[j], y2[j]), 255, 1)

            start = points[i][0], points[i][1]
            end = points[(i + 1) % len(points)][0], points[(i + 1) % len(points)][1]

            cv2.line(self.img, start, end, color, 3)

        if grayLevels:
            if self.currentToothModel.name > 4:
                img = self.currentRadiograph.imgLowerJaw
            else:
                img = self.currentRadiograph.imgUpperJaw
            if self.currentToothModel.name == -1:
                img = self.currentRadiograph.imgPyramid[0]
            profilesForLandmarkPoints = landmark.getGrayLevelProfilesForNormalPoints(
                img=img,
                sampleAmount=self.currentToothModel.sampleAmount,
                grayLevelModelSize=self.currentToothModel.sampleAmount,
                derive=False
            )

            for landmarkPointIdx in range(len(profilesForLandmarkPoints)):
                for profileContainer in profilesForLandmarkPoints[landmarkPointIdx]:
                    grayLevelProfile = profileContainer["grayLevelProfile"]
                    grayLevelProfilePoints = profileContainer["grayLevelProfilePoints"]
                    for z in range(len(grayLevelProfilePoints)):
                        start = (int(grayLevelProfilePoints[z][0]), int(grayLevelProfilePoints[z][1]))

                        cv2.line(self.img, start, start, int(grayLevelProfile[z]), thickness=2)

    def preprocessCurrentRadiograph(self):
        if not self.preprocess:
            self.img = images.preprocessRadiographImage(self.img)
        else:
            self.refreshCurrentImage()

        self.preprocess = not self.preprocess

    def findBetterLandmark(self):
        """
        Execute an iteration of "matching model points to target points"
        model points are defined by the model, target points by the 'bestLandmark'
        """
        previousLandmark = self.currentLandmark
        d = float("inf")
        i = 0

        if self.currentToothModel.name <= 4:
            jawImg = self.currentRadiograph.imgUpperJaw
        else:
            jawImg = self.currentRadiograph.imgLowerJaw

        if self.currentToothModel.name == -1:
            jawImg = self.currentRadiograph.imgPyramid[0]

        while i < 10 and d > 10:
            newTargetPoints = self.currentToothModel.findBetterFittingLandmark(jawImg, previousLandmark)
            improvedLandmark = self.currentToothModel.matchModelPointsToTargetPoints(newTargetPoints)

            d = improvedLandmark.shapeDistance(previousLandmark)
            previousLandmark = improvedLandmark

            i += 1
            print("Improvement iteration {}, distance {}".format(i, d))

        self.currentLandmark = previousLandmark
        self.refreshCurrentImage()
        self.drawLandmark(self.currentLandmark, 255)

    def showLowerJaw(self):
        self.img = self.currentRadiograph.imgLowerJaw.copy()

    def showUpperJaw(self):
        self.img = self.currentRadiograph.imgUpperJaw.copy()

    def setIncisorModel(self):
        self.currentToothModel = self.incisorModels[1]

    def getAllToothCenters(self):
        c = {}
        for i, model in enumerate(self.toothModels):
            center = model.initializationModel.meanCenter
            c[i] = int(center[0]), int(center[1])
        return c

    def drawToothCenters(self):
        for pos in self.toothCenters.values():
            cv2.circle(self.img, pos, 1, 255, 5)

    def drawGreyLevelProfile(self, points, profile):
        for i, point in enumerate(points):
            cv2.circle(self.img, point, 1, int(profile[i]), 2)

    def findBetterToothCenters(self):
        oldCenters = self.toothCenters

        converged = False
        cumulativeDistance = float("inf")

        while not converged:
            previousDistance = cumulativeDistance
            cumulativeDistance = 0

            for i, m in enumerate(self.toothModels):
                currentCenterForModel = oldCenters[i]
                distance, newCenter = m.initializationModel.getBetterCenter(
                    img=self.currentRadiograph.imgPyramid[0],
                    currentCenter=currentCenterForModel
                )
                self.toothCenters[i] = newCenter
                cumulativeDistance += distance

            print("center improvement iteration cumulative distance:", cumulativeDistance)
            converged = previousDistance - cumulativeDistance < 0.01

        print("centers converged")

        self.refreshCurrentImage()

    def initIncisorModels(self, modelNr=0):
        _, x = self.img.shape
        self.currentToothModel = self.incisorModels[modelNr]

        self.currentLandmark = self.incisorModels[modelNr].initLandmark(self.meanSplitLine, x)
        self.drawLandMarkWithNormals(self.currentLandmark, grayLevels=True)
        # centers = self.incisorModels[model].getCentersOfInitModel(self.currentLandmark)

        # for c in centers:
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

        # Show the result
        self.drawLandmark(self.currentLandmark, 255)

    def getMaskForRadiograph(self):
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
