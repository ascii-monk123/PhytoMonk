import numpy as np
import matplotlib.pyplot as plt
import cv2
from ..utils.helpers import stretch_contrast, smoothen_image, extract_channel
from typing import List
from skimage.filters import threshold_multiotsu

#extract background from given input image
def extract_background(image):
    '''
    Perform k-means channel segmentation on the input spot affected leaf.
    
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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    K=3
    attempts=20
    _,label,center=cv2.kmeans(vector, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
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
    b_channel = resArr[:,:,2]
    
    minVal = np.amin(b_channel)
    resImg = []

    #loop and apply thresholding
    for ele1 in resArr.reshape(-1,3):
        b = ele1[2]
        if b == minVal:
            resImg.append([0])
        else:
            resImg.append([255])
    
    resImg = np.array(resImg, dtype = np.uint8).reshape(image.shape[:2])
    opening = cv2.morphologyEx(resImg, cv2.MORPH_OPEN, kernel = np.ones((5, 5)), iterations = 2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel = np.ones((5, 5)), iterations = 2)
    cnts, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #fill contour for mask
    mask = np.ones(im.shape[:2], dtype="uint8") * 0
    mask = cv2.drawContours(mask, cnts, -1, (255), cv2.FILLED)
    return mask

#extract leaf using mask
def use_mask(mask:np.ndarray, image:np.ndarray, invert:bool = False) -> np.ndarray:
    '''|
    Perform and operation on input image and mask
    Inputs:
    mask => binary mask 
    image => np.ndarray representation of the image
    invert => boolean, if set to true, mask will be inverted before pixelwise & operation.
    Returns:
    resimage => np.ndarray representation of the and operation between mask and image.
    '''
    #if we want background
    if invert is True:
        mask = 255 - mask
        
    resimage = cv2.bitwise_and(image, image, mask = mask)
    resimage[mask == 0] = [0, 0, 0]
    return resimage


#first technique for pepper and tomato bacterial spot extraction
def k_means_cluster(leaf_image, leaf_mask, k_val):
    '''
    Perform a* channel segmentation on the input bacterial spot affected leaf.
    
    Inputs:
    leaf_image=> np.ndarray representation of the image. Background must be segmented
    leaf_mask=> mask for the leaf region on the image.
    k_val => number of clusters required
    
    returns:
    mask_s => segmentation result 
    '''
    shape=leaf_image.shape
    lm = cv2.cvtColor(leaf_image, cv2.COLOR_RGB2LAB)
    vector=lm[:,:,1:2]
    vector=vector.reshape(-1,1)
    #convert to float32
    vector=np.float32(vector)
    #criteria
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,1.0)
    #k-means clustering
    K=k_val
    attempts=10
    ret,label,center=cv2.kmeans(vector,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center=np.uint8(center)
    res=center[label.flatten()]
    #store cluster results
    resArr=[]
    for ele in res:
        l,a,b=90,ele[0],90
        resArr.append([l, a, b])
    resArr=np.array(resArr,dtype=np.uint8)
    resArr=resArr.reshape(shape)
    #extract a* channel
    a_channel = resArr[:, :, 1]
    mask_s = []
    maxi = np.amax(a_channel)
    for pixel in a_channel.reshape(-1):
        if pixel == maxi:
            mask_s.append([255])
            continue
        mask_s.append([0])
    #post processing
    mask_s = np.array(mask_s, dtype = np.uint8).reshape(a_channel.shape)
    mask_s = cv2.bitwise_and(mask_s, mask_s, mask = leaf_mask)
    return mask_s

#special method for peach bacterial spot extraction
def peach_special_extraction(leaf_image:np.ndarray, mask_s:np.ndarray):
    '''
    Inputs:
    leaf_image => original background segmented leaf image
    mask_s => bacterial spot kmeans segmentation mask
    Returns:
    resArr => final segmented lesion mask
    '''
    #convert leaf image to grayscale
    gray = cv2.cvtColor(leaf_image, cv2.COLOR_RGB2GRAY)
    #contrast stretching
    con_gray = stretch_contrast(gray)
    #get manual threshold mask
    _, th = cv2.threshold(con_gray, 128, 255, cv2.THRESH_BINARY)
    th = 255-th
    #multiotsu thresholding
    thresholds = threshold_multiotsu(con_gray)
    regions = np.digitize(con_gray, bins=thresholds)
    #get the reguired lesion regions
    maxi = np.amax(regions)
    resArr = []

    for pix in regions.reshape(-1):
        if pix == maxi:
            resArr.append(0)
        else:
            resArr.append(255)       
    resArr = np.array(resArr, dtype = np.uint8).reshape(regions.shape)
    #remove excess yellow and orange from leaf region
    resArr = cv2.bitwise_and(resArr, resArr, mask = th)
    resArr = cv2.bitwise_and(resArr, resArr, mask = mask_s)

    return resArr


