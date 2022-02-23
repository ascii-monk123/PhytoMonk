import cv2
import numpy as np
import matplotlib.pyplot as plt
from helpers import load_image, smoothen_image, use_mask
import detection.mildew as mld
import detection.rust as rst
import sys
import argparse

#main function
def main(path, leaf_type):

    #load image
    image = load_image(path)
    #perform gaussian smoothing
    smooth = smoothen_image(image)

    if leaf_type.lower() =="rust":
        #perform background extraction
        leaf_mask = rst.extract_background(image)
        print("Plotting the leaf mask")
        fig = plt.figure()
        plt.title("Leaf mask")
        plt.imshow(leaf_mask, cmap = 'gray')
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close(fig)

        #use leaf mask on iage
        leaf = use_mask(leaf_mask, smooth)
        print("Applying leaf mask...")
        fig = plt.figure()
        plt.title("Leaf Mask")
        plt.imshow(leaf, cmap = 'gray')
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close(fig)

        #perform thresholding on a* channel
        segment = rst.segment_disease(leaf_image = leaf, l_mask = leaf_mask, typ = "a")
        fig, axs = plt.subplots(1, 2)
        #cv2.bitwise_and(image,image, mask = segment)
        axs[0].imshow(cv2.bitwise_and(image, image, mask = segment), cmap = 'gray')
        axs[1].imshow(image)
        axs[0].set_title("Segmented leaf a* thresholding")
        axs[1].set_title("Original leaf")
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close(fig)

        #perform thresholding on h channel
        segment_h = rst.segment_disease(leaf, leaf_mask.copy(), "h")
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(cv2.bitwise_and(image,image, mask = segment_h), cmap = 'gray')
        axs[1].imshow(image)
        axs[0].set_title("Segmented leaf h thresholding")
        axs[1].set_title("Original leaf")
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close(fig)

        sys.exit()       


    #for mildew
    #perform background extraction
    leaf_mask = mld.extract_background(smooth)
    print("Plotting the leaf mask.....")
    fig = plt.figure()
    plt.title("Leaf Mask")
    plt.imshow(leaf_mask, cmap = 'gray')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    #use mask on image
    leaf = use_mask(leaf_mask, image)
    print("Appplying the leaf mask.....")
    fig = plt.figure()
    plt.title("Result of leaf mask")
    plt.imshow(leaf)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    #perform thresholding using b* channel
    segment = mld.segment_disease_b(leaf, leaf_mask)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(cv2.bitwise_and(image,image, mask = segment), cmap = 'gray')
    axs[1].imshow(image)
    axs[0].set_title("Segmented leaf b* thresholding")
    axs[1].set_title("Original leaf")
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    #plot b* detected region

    cache = mld.plot_detected_region(segment, image)
    f, axs = plt.subplots(1, 3)
    f.set_figheight(15)
    f.set_figwidth(15)
    axs[0].set_title("Detected region")
    axs[1].set_title("Segmented region")
    axs[2].set_title("Original image")
    axs[0].imshow(cache[0])
    axs[1].imshow(cache[1])
    axs[2].imshow(image)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(f)


    #perform k-means segmentation
    print("Performing kmeans segmentation...")
    segmented = mld.k_means_seg(leaf, leaf_mask)
    plt.imshow(segmented, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title("Kmeans segmentation result")
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    #plot kmeans detected region
    cache = mld.plot_detected_region(segmented, image, None, method = "KM")
    f, axs = plt.subplots(1, 3)
    f.set_figheight(15)
    f.set_figwidth(15)
    axs[0].set_title("Detected region")
    axs[1].set_title("Segmented region")
    axs[2].set_title("Original image")
    axs[0].imshow(cache[0])
    axs[1].imshow(cache[1])
    axs[2].imshow(image)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(f)


#/home/aahan/Documents/Major Project/mildew/cherry_powdery_mildew/31.jpg --imagepath
#parse arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", action = "store", dest = "path")
    parser.add_argument("-t", action = "store", dest="type")

    #read image
    results = parser.parse_args()
    main(results.path, results.type)

    
