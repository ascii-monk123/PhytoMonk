# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import cv2
import os
from .utils.helpers import smoothen_image, use_mask, plot_detected_region
from django.http import HttpResponse
from wsgiref.util import FileWrapper
from PIL import Image
from .detection import mildew as mld
from .detection import rust as rst
import io

#convert numpy array to iamge
def to_image(numpy_img):
    img = Image.fromarray(numpy_img, 'RGB')
    return img

#extract lesions from image
def detection(image, disease_type, smooth):
	#if rust disease
	if disease_type.lower() == "rust":
		#perform rust background extraction
		leaf_mask = rst.extract_background(smooth)
		#use leaf_mask to extract leaf region
		leaf = use_mask(leaf_mask, smooth)

		#perform a* thresholding
		segment_a = rst.segment_disease(leaf, l_mask = leaf_mask, typ = "a")
		#get lesion regions and segmented binary image
		cache_a = plot_detected_region(segment_a, image)

		#perform h thresholding
		segment_h = rst.segment_disease(leaf, leaf_mask.copy(), "h")
		#get lesion region for h channel thresholds
		cache_h = plot_detected_region(segment_h, image)

		return [cache_a, cache_h]
	
	#if mildew disease
	if disease_type.lower() == "mildew":
		#get the leaf_mask
		leaf_mask = mld.extract_background(smooth)
		#use leaf_mask to extract leaf region
		leaf = use_mask(leaf_mask, smooth)

		#perform b* thresholding
		segment_b = mld.segment_disease_b(leaf, leaf_mask)
		#get lesion region for mildew b* thresholding
		cache_b = plot_detected_region(segment_b, image)

		#perform k-means thresholding
		segment_km = mld.k_means_seg(leaf, leaf_mask)
		cache_k = plot_detected_region(segment_km, image)

		return [cache_b, cache_k]




@csrf_exempt
def detect(request):
	# initialize the data dictionary to be returned by the request
	data = {"success": False}
	image = None
	#is it post request
	if request.method == "POST":
		# check if image was uploaded
		if request.FILES.get("image", None) is not None:
			# grab the uploaded image
			image = _grab_image(stream=request.FILES["image"])

		# otherwise assume a URL was passed
		else:
			# grab the URL from the request
			url = request.POST.get("url", None)

			# if the URL is None, then return an error
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)

			# load the image and convert
			image = _grab_image(url=url)

		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#apply gaussian blurring
		smooth = smoothen_image(image)

		#check which disease type to detect
		if request.POST.get("disease_type", None) is not None:
			disease_type = request.POST.get("disease_type")
			#call the detection technique
			cache = detection(image, disease_type, smooth)
			#if cache length is 2 loop through and generate images
			if len(cache) == 2:
				test = to_image(np.array(cache[0][0]).astype(np.uint8))
				file_like_object = io.BytesIO()
				test.save(file_like_object, format='png')
				response = HttpResponse(file_like_object.getvalue(), content_type = "image/png")
				response['Content-Disposition'] = 'attachment; filename="result_lol.png"'

				return response

		else:
			data["error"] = "No infection specified."
			return JsonResponse(data)

		


	

	# return a JSON response
	return JsonResponse(data)

def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)

	# otherwise, the image does not reside on disk
	else:	
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.urlopen(url)
			data = resp.read()

		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()

		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image