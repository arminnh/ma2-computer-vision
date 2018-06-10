## Based on code from: https://github.com/Cartucho/OpenLabeling

import glob

import cv2
import numpy as np

WITH_QT = True
try:
    cv2.namedWindow("Test")
    cv2.displayOverlay("Test", "Test QT", 1000)
except:
    WITH_QT = False
cv2.destroyAllWindows()

img_index = 0
img = None
mouse_x = 0
mouse_y = 0

def changeImgIndex(x):
    global img_index, img
    img_index = x
    img_path = image_list[img_index]
    img = cv2.imread(img_path)
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, "Showing image "
                                        "" + str(img_index) + "/"
                                                              "" + str(last_img_index), 1000)
    else:
        print("Showing image "
              "" + str(img_index) + "/"
                                    "" + str(last_img_index) + " path:" + img_path)

def drawEdges(tmp_img):
    blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
    edges = cv2.Canny(blur, 0, 250, 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Overlap image and edges together
    tmp_img = np.bitwise_or(tmp_img, edges)
    # tmp_img = cv2.addWeighted(tmp_img, 1 - edges_val, edges, edges_val, 0)
    return tmp_img

def decreaseIndex(current_index, last_index):
    current_index -= 1
    if current_index < 0:
        current_index = last_index
    return current_index


def increaseIndex(current_index, last_index):
    current_index += 1
    if current_index > last_index:
        current_index = 0
    return current_index


# mouse callback function
def mouseListener(event, x, y, flags, param):
    global mouse_x, mouse_y

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
    elif event == cv2.EVENT_LBUTTONDOWN:
        print("click x: {}, y: {}".format(mouse_x, mouse_y))


def getImageList():
    img_dir = "../resources/data/radiographs/"
    image_list = glob.glob(img_dir + '*.tif')
    image_list.extend(glob.glob(img_dir + "/extra/*.tif"))
    return image_list


# load img list
image_list= getImageList()
# print(image_list)
last_img_index = len(image_list) - 1

# create window
WINDOW_NAME = 'Computer Vision KU Leuven'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_NAME, 1000, 700)
cv2.setMouseCallback(WINDOW_NAME, mouseListener)

# selected image
TRACKBAR_IMG = 'Image'

cv2.createTrackbar(TRACKBAR_IMG, WINDOW_NAME, 0, last_img_index, changeImgIndex)

# initialize
changeImgIndex(0)
edges_on = False

# loop
while True:
    # clone the img
    tmp_img = img.copy()
    height, width = tmp_img.shape[:2]
    if edges_on == True:
        # draw edges
        tmp_img = drawEdges(tmp_img)
    img_path = image_list[img_index]
    cv2.imshow(WINDOW_NAME, tmp_img)
    pressed_key = cv2.waitKey(50)

    """ Key Listeners START """
    if pressed_key == ord('a') or pressed_key == ord('z'):
        # show previous image key listener
        if pressed_key == ord('a'):
            img_index = decreaseIndex(img_index, last_img_index)
        # show next image key listener
        elif pressed_key == ord('z'):
            img_index = increaseIndex(img_index, last_img_index)
        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)
    # show edges key listener
    elif pressed_key == ord('e'):
        if edges_on == True:
            edges_on = False
            if WITH_QT:
                cv2.displayOverlay(WINDOW_NAME, "Edges turned OFF!", 1000)
            else:
                print("Edges turned OFF!")
        else:
            edges_on = True
            if WITH_QT:
                cv2.displayOverlay(WINDOW_NAME, "Edges turned ON!", 1000)
            else:
                print("Edges turned ON!")

    # quit key listener
    elif pressed_key == ord('q'):
        break
    """ Key Listeners END """

    if WITH_QT:
        # if window gets closed then quit
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

cv2.destroyAllWindows()
