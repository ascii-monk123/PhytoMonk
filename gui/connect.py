import requests
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import zlib
from quantify import quantify
from typing import List

#utility function to generate results
def plot_detected_region(segment:np.ndarray, image:np.ndarray) -> List :
    '''
    Utility function to plot infected region on original leaf image
    Inputs:
    segment => segmentation result
    image => original leaf image 
    returns:
    cache => List containing detected region image.

    '''
    cpy = image.copy()
    #plot red infected region
    cpy[segment == 255] = [255, 0, 0]

    return cpy
    

#decompress nparray
def uncompress_nparr(bytestring):
    """
    """
    return np.load(io.BytesIO(zlib.decompress(bytestring)))


#make requests to server
def get_results(server_url:str, imagePath:str, disease_type:str):
    url = server_url
    files = {'image' : open(imagePath, 'rb')}

    values = {'disease_type': disease_type}

    #make http request
    res = requests.post(url, files = files, data = values)
    #extract resultant images
    data = uncompress_nparr(res.content)
    #load image
    image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
    #segments
    segment_1 = np.array(data[0], dtype = np.uint8)
    segment_2 = np.array(data[1], dtype = np.uint8)
    #plotted image
    plot_1 = plot_detected_region(segment_1, image)
    plot_2 = plot_detected_region(segment_2, image)
    leaf_im = np.array(data[2], dtype = np.uint8)
    #quant results
    quant_a = quantify(leaf_im, segment_1)
    quant_b = quantify(leaf_im, segment_2)

    
    return (plot_1, plot_2, quant_a, quant_b)




