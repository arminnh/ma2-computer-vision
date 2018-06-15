import images
import util
from preprocess_img import *


class GUI:

    def __init__(self, radiographs, models, incisorModels):
        self.name = "Computer Vision KU Leuven"
        self.radiographs = radiographs
        self.currentRadiograph = None
        self.currentRadiographIndex = 0
        self.toothModels = models
        self.currentToothModel = None
        self.currentToothModelIndex = 0
        self.toothCenters = self.getAllOrigins()
        self.incisorModels = incisorModels
        self.img = None
        self.preprocess = False
        self.currentLandmark = None
        self.showEdges = False
        self.c = 0
        self.blockSize = 3

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
            cv2.line(img, (int(x / 2), 0), (int(x / 2), int(y)), (255, 255, 255), 3)
            cv2.line(img, (0, int(y / 2)), (x, int(y / 2)), (255, 255, 255), 3)

            # for model in self.toothModels:
            #     if self.currentRadiographIndex < len(model.landmarks):
            #         landmark = model.landmarks[self.currentRadiographIndex]
            #         self.drawLandmark(landmark, color=180, thickness=3)
            #
            #         self.drawOriginModel(model.initializationModel.profileForImage[self.currentRadiographIndex])
            #         self.drawAllOrigins()

            cv2.imshow(self.name, img)

            # Key Listeners
            pressed_key = cv2.waitKey(50)

            if pressed_key == ord("`"):
                self.c = 0
                self.blockSize = 3

            if pressed_key == ord("1"):
                self.img = cv2.medianBlur(self.img, 29)

            if pressed_key == ord("2"):
                self.c += 1
                print(self.c)
                self.refreshCurrentImage()
                self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                 self.blockSize, self.c)

            if pressed_key == ord("3"):
                self.blockSize += 2
                print(self.blockSize)
                self.refreshCurrentImage()
                self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                 self.blockSize, self.c)

            if pressed_key == ord("4"):
                self.c += 1
                print(self.c)
                self.refreshCurrentImage()
                self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                 self.blockSize, self.c)

            if pressed_key == ord("5"):
                self.blockSize += 2
                print(self.blockSize)
                self.refreshCurrentImage()
                self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                 self.blockSize, self.c)

            if pressed_key == ord("6"):
                self.refreshCurrentImage()
                self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45,
                                                 5)

            if pressed_key == ord("x") or pressed_key == ord("c"):
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
                    cv2.displayOverlay(self.name, "Edges turned OFF!", 100)

                else:
                    self.showEdges = True
                    cv2.displayOverlay(self.name, "Edges turned ON!", 100)

            elif pressed_key == ord("p"):
                self.preprocessCurrentRadiograph()

            # quit key listener
            elif pressed_key == ord("q"):
                break

            elif pressed_key == ord("o"):
                # self.showOriginalTooth()
                self.updateOrigins()

            elif pressed_key == ord("n"):
                self.findBetterLandmark()

            elif pressed_key == ord("u"):
                self.showUpperJaw()

            elif pressed_key == ord("l"):
                self.showLowerJaw()

            elif pressed_key == ord("w"):
                self.crazyLineDetector()

            elif pressed_key == ord("s"):
                self.setIncisorModel()

            if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

    def getAllOrigins(self):
        c = {}
        for i, model in enumerate(self.toothModels):
            orig = model.initializationModel.meanOrigin
            c[i] = int(orig[0]), int(orig[1])
        return c

    def drawAllOrigins(self):
        for pos in self.toothCenters.values():
            cv2.circle(self.img, pos, 1, int(200), 2)

    def drawOriginModel(self, originModel):
        for pixel, pos in originModel:
            orig = (int(pos[0]), int(pos[1]))
            cv2.circle(self.img, orig, 1, int(pixel), 2)

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

    def setCurrentImage(self, idx):
        self.currentRadiographIndex = idx
        self.currentRadiograph = self.radiographs[idx]
        self.toothCenters = self.getAllOrigins()
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
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
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
            print("click x: {}, y: {}".format(x, y))

            self.currentLandmark = self.currentToothModel.getTranslatedAndInverseScaledMean(x, y)

            self.drawLandMarkWithNormals(self.currentLandmark, grayLevels=True)

    def drawLandmark(self, landmark, color=(0, 0, 255), thickness=1):
        points = landmark.getPointsAsTuples().round().astype(int)

        for i in range(len(points)):
            origin = (points[i][0], points[i][1])
            end = (points[(i + 1) % len(points)][0], points[(i + 1) % len(points)][1])

            cv2.line(self.img, origin, end, color, thickness)

    def drawLandMarkWithNormals(self, landmark, grayLevels, color=(0, 0, 255)):
        points = landmark.getPointsAsTuples().round().astype(int)

        for i in range(len(points)):
            m = util.getNormalSlope(points[i - 1], points[i], points[(i + 1) % len(points)])
            p = np.asarray(util.sampleLine(m, points[i], self.currentToothModel.sampleAmount))
            x2 = p[:, 0]
            y2 = p[:, 1]

            for j in range(len(x2)):
                cv2.line(self.img,
                         (int(x2[j]), int(y2[j])),
                         (int(x2[j]), int(y2[j])),
                         (255, 0, 0), 1)

            origin = (int(points[i][0]), int(points[i][1]))
            end = (int(points[(i + 1) % len(points)][0]), int(points[(i + 1) % len(points)][1]))

            cv2.line(self.img, origin, end, color, 3)

        if grayLevels:
            if self.currentToothModel.name > 4:
                img = self.currentRadiograph.imgLowerJaw
            else:
                img = self.currentRadiograph.imgUpperJaw
            if self.currentToothModel.name == -1:
                img = self.currentRadiograph.img
            profilesForLandmarkPoints = landmark.getGrayLevelProfilesForNormalPoints(
                img=img,
                sampleAmount=self.currentToothModel.sampleAmount,
                derive=False
            )

            for landmarkPointIdx in range(len(profilesForLandmarkPoints)):
                for profileContainer in profilesForLandmarkPoints[landmarkPointIdx]:
                    grayLevelProfile = profileContainer["grayLevelProfile"]
                    grayLevelProfilePoints = profileContainer["grayLevelProfilePoints"]
                    print(grayLevelProfile)
                    for z in range(len(grayLevelProfilePoints)):
                        orig = (int(grayLevelProfilePoints[z][0]), int(grayLevelProfilePoints[z][1]))

                        cv2.line(self.img, orig, orig, int(grayLevelProfile[z]), thickness=2)

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
        d = 2
        i = 0

        correctHalf = self.currentRadiograph.imgUpperJaw if self.currentToothModel.name <= 4 else self.currentRadiograph.imgLowerJaw
        if self.currentToothModel.name == -1:
            correctHalf = self.currentRadiograph.img

        while d > 1 and i < 1:
            newTargetPoints = self.currentToothModel.findBetterFittingLandmark(previousLandmark,
                                                                               correctHalf)
            improvedLandmark = self.currentToothModel.matchModelPointsToTargetPoints(newTargetPoints)

            d = improvedLandmark.shapeDistance(previousLandmark)
            previousLandmark = improvedLandmark

            i += 1
            print("Improvement iteration {}, distance {}".format(i, d))

        self.currentLandmark = previousLandmark
        self.refreshCurrentImage()
        self.drawLandmark(self.currentLandmark, (255, 255, 255))

    def refreshCurrentImage(self):
        self.img = self.currentRadiograph.img.copy()

        jawSplitLine = self.currentRadiograph.jawSplitLine
        for i, (x, y) in enumerate(jawSplitLine):
            if i > 0:
                cv2.line(self.img, (jawSplitLine[i - 1][0], jawSplitLine[i - 1][1]), (x, y), 255, 2)

        self.meanSplitLine = int(np.mean(jawSplitLine[:,1]))
        cv2.line(self.img, (0,self.meanSplitLine), (self.img.shape[1],self.meanSplitLine), 200, 2)

    def updateOrigins(self):
        oldOrigins = self.toothCenters
        for i, m in enumerate(self.toothModels):
            currentOriginForModel = oldOrigins[i]
            correctHalf = self.currentRadiograph.imgUpperJaw if m.name <= 4 else self.currentRadiograph.imgLowerJaw

            newOrigin = m.initializationModel.getBetterOrigin(currentOriginForModel, correctHalf)
            newOrigin = (int(newOrigin[0]), int(newOrigin[1]))
            self.toothCenters[i] = newOrigin
        self.refreshCurrentImage()

    def showLowerJaw(self):
        self.img = self.currentRadiograph.imgLowerJaw.copy()

    def showUpperJaw(self):
        self.img = self.currentRadiograph.imgUpperJaw.copy()

    def crazyLineDetector(self):
        y, x = self.img.shape
        midX = int(x/2)

    def setIncisorModel(self):
        self.currentToothModel = self.incisorModels[2]



