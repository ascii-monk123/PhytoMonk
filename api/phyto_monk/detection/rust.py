import numpy as np
import matplotlib.pyplot as plt
import cv2
from ..utils.helpers import stretch_contrast, smoothen_image, extract_channel
from typing import List
from skimage.filters import threshold_multiotsu


def use_mask(mask:np.ndarray, image:np.ndarray, invert:bool = False) -> np.ndarray:
    '''
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

#kmeans clustering on h channel
def k_means_cluster_h(leaf_image, leaf_mask, k_val):
    shape=leaf_image.shape
    lm = cv2.cvtColor(leaf_image, cv2.COLOR_RGB2HSV)
    vector=lm[:,:,0:1]
    vector=vector.reshape(-1,1)

    #convert to float32
    vector=np.float32(vector)

    #criteria
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,1.0)
    K=k_val
    attempts=10
    ret,label,center=cv2.kmeans(vector,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)


    center=np.uint8(center)
    res=center[label.flatten()]

    resArr=[]
    for ele in res:
        h,s,v=ele[0],90,90
        resArr.append([h, s, v])
    resArr=np.array(resArr,dtype=np.uint8)
    resArr=resArr.reshape(shape)
    h_channel = resArr[:, :, 0]
    plt.imshow(h_channel, cmap  = 'gray')
    mask_s = []
    maxi = np.amax(h_channel)

    for pixel in h_channel.reshape(-1):
        if pixel == maxi:
            mask_s.append([0])
            continue
        mask_s.append([255])

    mask_s = np.array(mask_s, dtype = np.uint8).reshape(h_channel.shape)
    mask_s = cv2.bitwise_and(mask_s, mask_s, mask = leaf_mask)
    return mask_s
#function to binarize the lesions
def binarize(image):
    resArr = []
    for pixel in image.reshape(image.shape[0]*image.shape[1],-1):
        if pixel[0]==0 and pixel[1]==0 and pixel[2]==0:
            resArr.append(0)
        else:
            resArr.append(255)
    resArr = np.array(resArr, dtype = np.uint8).reshape(image.shape[0], image.shape[1])
    return resArr

#segmentation on the a* channel
def segment_disease(leaf_image:np.ndarray, l_mask:np.ndarray, typ:str = "a") -> np.ndarray:
    '''
    Perform a* channel segmentation on the input rust affected leaf.
    
    Inputs:
    image=> np.ndarray representation of the image. Background must be segmented
    mask=> mask for the leaf region on the image.
    type => which thresholding method to use (a for a* channel) (h for hsv channel)
    
    returns:
    resArr => segmentation result 
    '''

    if typ.lower()== "k_means_h":
        ms = k_means_cluster_h(leaf_image = leaf_image, leaf_mask= l_mask, k_val=4)
        fs = use_mask(ms, leaf_image)
        ms2 = k_means_cluster_h(fs, ms, 3)
        fms = use_mask(ms2, leaf_image)
        return binarize(fms)
    #convert to required colorspace and extract the channel
    #extract channel
    elif typ.lower() =="a":
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
    resArr = cv2.erode(resArr, kernel = np.ones((3, 3)), iterations = 2)


    return resArr

    






    