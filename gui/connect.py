import requests
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import zlib



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

    cache_a = np.array(data[0], dtype = np.uint8)
    cache_b = np.array(data[1], dtype = np.uint8)

    return (cache_a, cache_b)





