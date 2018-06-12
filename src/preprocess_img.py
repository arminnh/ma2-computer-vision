import cv2
from PIL import Image
import numpy as np

# Edge_Detection_and_Features_Extraction_f2.pdf
def equalizeHist(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def cvToPIL(img):
     return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def bilateralFilter(img):
    return cv2.bilateralFilter(img, 9, 100, 100)

def tophat(img):
    kernel = np.ones((130,130))
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

def showImg(img):
    cvToPIL(img).show()
    return img

def applyCLAHE(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2)
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ret3, th3 = cv2.threshold(blur, 100, 255,  cv2.THRESH_BINARY +  cv2.THRESH_OTSU)
    return cv2.cvtColor(th3, cv2.COLOR_GRAY2BGR)

def increaseContrast(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

def PILtoCV(img):
    img = np.array(img)
    # Change colors from RGB to BGR
    return img.copy()

