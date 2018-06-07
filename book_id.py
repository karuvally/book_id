#!/usr/bin/env python3
# Book ID, alpha release
# Copyright 2018, Dextro Labs

# import the essential stuff
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


# find contours, process them
def process_contours(edges):
    # find contours, sort them into list
    contours = cv2.findContours(edges.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            return approx


# pre process the image
def preprocess_image(image):
    ratio = image.shape[0] / 500.0
    resized_image = imutils.resize(image, height = 500)

    image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image, ratio


# the main function
def main():
    # manage runtime arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required = True,
        help = 'path to input image')
    arguments = vars(parser.parse_args())

     # load, preprocess image
    image = cv2.imread(arguments['image'])
    pre_image, ratio = preprocess_image(image)

    # find edges
    edges = cv2.Canny(pre_image, 75, 200) # optimize this

    # find contours of the page
    screen_contours = process_contours(edges)

    # debug
    out_image = cv2.cvtColor(pre_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(out_image, [screen_contours], -1, (0, 255, 0), 2)
    cv2.imshow('Outline', out_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)


# call the main function
main()
