import cv2
import numpy as np
import matplotlib.pyplot as plt

#function to load images
def load_image(path:str) -> np.ndarray:
    '''
    Loads image and automatically converts into rgb from bgr
    Inputs:
    path => string path from where to load the image
    Returns:
    image => numpy.ndarray representation of the image
    '''
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


#smoothen image
def smoothen_image(image:np.ndarray) -> np.ndarray:
    '''
    Performs gaussian blur on the input image.
    Inputs:
    image => numpy.nd array representation of the image
    returns:
    resultant image after gaussian blur
    '''
    return cv2.GaussianBlur(image, (5,5), 1)
    
#contrats stretching
def stretch_contrast(image:np.ndarray) -> np.ndarray:
    '''
    Function to perform contrast stretching given an input image.
    
    Inputs =>
    image : numpy.ndarray representation of the image (must be only one channel)
    
    Returns =>
    con_image : min-max contrast stretched representation of the input image
    
    '''
    
    con_image = np.copy(image).reshape(-1)
    
    #min intensity
    min_i = np.amin(con_image)
    #max intensity
    max_i = np.amax(con_image)
    
    for idx, pixel in enumerate(con_image):
        
        con_image[idx] = 255 * ((pixel - min_i) / (max_i - min_i))
    
    #reshape and return
    con_image = con_image.reshape(image.shape)
    con_image = con_image.astype("uint8")
    
    return con_image

#use mask on an image
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

#extract channel
def extract_channel(colorspace:str, channel_idx:int, image:np.ndarray) -> np.ndarray:
    '''
    Extracts color-channel from given image.

    Inputs=>
    colorspace: either rgb or hsv. If invalid, return image[:, :, channel_idx]
    channel_idx: index of the channel needed
    image: np.ndarray representation of the image in rgb

    Returns=>
    channel: np.ndarray color channel representation
    '''
    color = None
    if colorspace.lower() == 'hsv':
        color = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    elif colorspace.lower()=='lab':
        color = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    else:
        color = image.copy()
    
    return color[:, :, channel_idx]