''' Cell counting '''


import cv2
import numpy as np


def imshow(name, images):
    ''' Display images (a list with images all equal number of channels) all together '''
    image = np.concatenate(images, axis=1)
    image = cv2.resize(image, dsize=tuple([s // 2 for s in image.shape if s > 3])[::-1])
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    
def detect_edges(img):
    ''' Canny edge detection '''  
    ########
    ##TODO##
    ########
    imshow('Processed Images', [grayscale_img, edges_binary, edges_on_grayscale_img])
    
    
def detect_circles(img):
    ''' Hough transform to detect circles ''' 
    ########
    ##TODO##
    ########
    imshow('All Detected Circles', [circles_on_original_img])
    return hough_transform
        
    
def calculate_features(img, circles):
    ''' Use the Hough transform to derive a feature vector for each circle ''' 
    ########
    ##TODO##
    ########
    return img_features


def threshold_circles(img, circles, features, thresholds):
    ''' Threshold the feature vector to get the "right" circles ''' 
    ########
    ##TODO##
    ########
    imshow('Only Selected Circles', [selected_circles_on_original_image])
    return n
       
 
if __name__ == '__main__':
    
    #read the image
    img = cv2.imread('normal.jpg')
    
    #show the image
    imshow('Original Image', [img])
    
    #do detection
    detect_edges(img)
    circles = detect_circles(img)
    features = calculate_features(img, circles)
    n = threshold_circles(img, circles, features, ((t1, T1), (t2, T2), (t3, T3)))

    #print result
    print("We counted {} cells.".format(n))