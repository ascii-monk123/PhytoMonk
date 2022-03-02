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



url = 'http://127.0.0.1:8000/detect/'

files = {'image' : open('/home/aahan/Documents/Major Project/mildew/cherry_powdery_mildew/54.jpg', 'rb')}

values = {'disease_type': 'mildew'}

#make http request

res = requests.post(url, files = files, data = values)


data = uncompress_nparr(res.content)

cache_a = data[0]
cache_b = data[1]

first = cache_b[0]

first = np.array(first, dtype = np.uint8)

plt.imshow(first)
plt.show()


