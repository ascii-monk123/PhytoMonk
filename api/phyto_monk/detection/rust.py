import numpy as np
import matplotlib.pyplot as plt
import cv2
from ..utils.helpers import stretch_contrast, smoothen_image, extract_channel
from typing import List
from skimage.filters import threshold_multiotsu



def extract_background(image:np.ndarray) -> np.ndarray:
    '''
    Given an input image, extract the background from them. Works only for plant village rust images as of now.
    Inputs:
    image => numpy.ndarray representation of the image
    Returns:
    binary mask for subtracting leaf from the background
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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    K=2
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
    for ele1 in resArr.reshape(-1,3):

        a = ele1[1]

        if a == minVal:
            resImg.append([255])

        else:
            resImg.append([0])
    #post-processing
    resImg = np.array(resImg, dtype = np.uint8).reshape(image.shape[:2])
    closing = cv2.morphologyEx(resImg, cv2.MORPH_CLOSE, kernel = np.ones((5, 5)), iterations = 2)
    cnts, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #fill contour for mask
    mask = np.ones(im.shape[:2], dtype="uint8") * 0
    mask = cv2.drawContours(mask, cnts, -1, (255), cv2.FILLED)
    return mask



#extract disease after multi otsu
def extract_disease_multi_otsu(regions:np.ndarray, typ:str, mask:np.ndarray = None) -> np.ndarray:
    '''
    Extract diseased region after multi-otsu.
    Inputs=>
    region: np.ndarray representation of multiple otsu on diseased leaf
    type: str (a means a* thresholding) (h means h thresholding)
    mask: leaf mask.specify if h thresholding used
    Returns=>
    resArr: np.array result of extracting diseased region

    '''
    maxi = np.amax(regions)
    regions = cv2.bitwise_and(regions, regions, mask = mask)
    resArr = []

    #a* thresholding
    if typ.lower() == 'a':
        #pixel wise loop for extracting diseased regions
        for pix in regions.reshape(-1):
            if pix == maxi:
                resArr.append(255)
            else:
                resArr.append(0)

        resArr = np.array(resArr, dtype = np.uint8).reshape(regions.shape)
        resArr = cv2.bitwise_and(resArr, resArr, mask = mask)

    #h thresholding
    elif typ.lower() == "h":
        for pix in regions.reshape(-1):
            if pix == maxi:
                resArr.append(255)
            else:
                resArr.append(0)

        resArr = np.array(resArr, dtype = np.uint8).reshape(regions.shape)
        resArr = 255 - resArr
        resArr = cv2.bitwise_and(resArr, resArr, mask = mask)

    return resArr

#segmentation on the a* channel
def segment_disease(leaf_image:np.ndarray, l_mask:np.ndarray, typ:str = "a") -> np.ndarray:
    '''
    Perform a* channel segmentation on the input mildew affected leaf.
    
    Inputs:
    image=> np.ndarray representation of the image. Background must be segmented
    mask=> mask for the leaf region on the image.
    type => which thresholding method to use (a for a* channel) (h for hsv channel)
    
    returns:
    resArr => segmentation result 
    '''
    #convert to required colorspace and extract the channel
    #extract channel
    if typ.lower() =="a":
        channel = extract_channel('lab', 1, leaf_image)
    elif typ.lower() =="h":
        channel = extract_channel('hsv', 0, leaf_image)

    channel = cv2.bitwise_and(channel, channel, mask = l_mask.copy())
    #stretch contrast
    con_ch = stretch_contrast(channel)

    #perform otsu multithreshold
    thresholds = threshold_multiotsu(con_ch)
    regions = np.digitize(con_ch, bins=thresholds)

    #cleanup after otsu multithreshold
    resArr = extract_disease_multi_otsu(regions, typ, l_mask.copy())


    return resArr

    






    