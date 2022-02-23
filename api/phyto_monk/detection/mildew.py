import numpy as np
import matplotlib.pyplot as plt
import cv2
from ..utils.helpers import stretch_contrast, smoothen_image
from typing import List

#function to extract background from the image
def extract_background(image:np.ndarray) -> np.ndarray:
    '''
    Given an input image, extract the background from them. Works only for plant village mildew images as of now.
    Inputs:
    image => numpy.ndarray representation of the image
    Returns:
    binary mask for subtracting leaf from the background
    '''
    #extract a* channel
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    a = lab[:, :, 1] 
    
    #contrast stretching
    con_a = stretch_contrast(a)
    _, th = cv2.threshold(con_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = 255 - th
    
    #filling holes inside image
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel = np.ones((5, 5)), iterations = 2)
    cnts, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #fill contour for mask
    mask = np.ones(image.shape[:2], dtype="uint8") * 0
    mask = cv2.drawContours(mask, cnts, -1, (255), cv2.FILLED)
    
    return mask
    
#lab segmentation method
def segment_disease_b(image:np.ndarray, mask:np.ndarray) -> np.ndarray:
    '''
    Perform b* channel segmentation on the input mildew affected leaf.
    
    Inputs:
    image=> np.ndarray representation of the image. Background must be segmented
    mask=> mask for the leaf region on the image.
    
    returns:
    segment => segmentation result 
    '''
    #smoothen and extract b channel
    smooth = smoothen_image(image)
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    b = lab[:, :, 2]
    
    
    #use mask to make background black
    b[mask == 0] = [0]
    
    #stretch contrast
    con_b = stretch_contrast(b)

    #apply otsu thresholding
    _, segment = cv2.threshold(con_b, 220, 255, cv2.THRESH_BINARY)
    segment = 255 - segment

    #morphological opening
    segment = cv2.morphologyEx(segment, cv2.MORPH_OPEN, kernel = np.ones((5, 5)), iterations = 1)
    segment = cv2.erode(segment, kernel = np.ones((5, 5)), iterations = 1)
    segment = cv2.dilate(segment, kernel = np.ones((3, 3)), iterations = 1)

    
    #maks background black
    segment[mask == 0] = [0]

    return segment

#kmeans segmentation method
def k_means_seg(image, mask):
    '''
    Perform k-means channel segmentation on the input mildew affected leaf.
    
    Inputs:
    image=> np.ndarray representation of the image. Background must be segmented
    mask=> mask for the leaf region on the image.
    
    returns:
    segment => segmentation result 
    '''
    im = image.copy()
    #convert to lab
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)

    #create a vector containing a*b* channel
    shape = lab.shape
    vector = lab[:, :, 1:]
    vector = vector.reshape(-1, 2)

    #convert to float32
    vector = np.float32(vector)

    #apply kmeans algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K=3
    attempts=10
    _,label,center=cv2.kmeans(vector, K, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    #get centroids
    center = np.uint8(center)
    res = center[label.flatten()]

    resArr = []
    #convert into an image with constant luminance
    for ele in res:
        a,b = ele[0], ele[1]
        resArr.append([90, a, b])
        
    resArr = np.array(resArr, dtype = np.uint8).reshape(shape)

    #segment automatically based on min a* value
    a_channel = resArr[:,:,1]
    
    minVal = np.amin(a_channel)

    resImg = []

    #loop and apply thresholding
    for ele1, ele2 in zip(resArr.reshape(-1,3), im.reshape(-1,3)):

        a = ele1[1]

        if a == minVal:
            resImg.append([0, 0, 0])

        else:
            resImg.append([ele2[0], ele2[1], ele2[2]])
            

    resImg = np.array(resImg, dtype = np.uint8).reshape(shape)
    a = resImg[:, :, 1]
    
    _, thres = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    thres = cv2.erode(thres, kernel = np.ones((3, 3)), iterations = 1)
    thres = cv2.dilate(thres, kernel = np.ones((3, 3)), iterations = 1)
    
    return thres
