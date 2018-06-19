#!/usr/bin/env python3
# Book ID, alpha release
# Copyright 2018, Dextro Labs

# TODO
# Try normal cannny edge
# Other edge detection algorithms

# import the essential stuff
import numpy as np
import argparse
import cv2
import imutils

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours


# display image for debugging
def display_image(image):
    cv2.imshow('Output', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# transform, thanks to Adrian :D
def four_point_transform(image, points):
    rectangle = np.zeros((4, 2), dtype = 'float32')
    sum_of_points = points.sum(axis = 1)
    rectangle[0] = points[np.argmin(sum_of_points)]
    rectangle[2] = points[np.argmax(sum_of_points)]

    difference = np.diff(points, axis = 1)
    rectangle[1] = points[np.argmin(difference)]
    rectangle[3] = points[np.argmax(difference)]
    (tl, tr, br, bl) = rectangle

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    destination_points = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype = 'float32')

    transform_matrix = cv2.getPerspectiveTransform(rectangle,
        destination_points)
    
    warped_image = cv2.warpPerspective(image, transform_matrix,
        (max_width, max_height))

    return warped_image


# automatic canny edge detection
def auto_canny(image):
    sigma = 0.50 # tweak this, default = 0.33
    image_median = np.median(image)

    lower_threshold = int(max(0, (1.0 - sigma) * image_median))
    upper_threshold = int(min(255, (1.0 + sigma) * image_median))
    
    edges = cv2.Canny(image, lower_threshold, upper_threshold)

    return edges


# find contours, process them
def process_contours(edges):
    # find contours, sort them into list
    contours = cv2.findContours(edges.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1]
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

    # convert image to binary
    #_, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

    return image, ratio


# the main function
def main():
    # essential variables
    distance_list = []

    # manage runtime arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required = True,
        help = 'path to input image')
    arguments = vars(parser.parse_args())

     # load, preprocess image
    image = cv2.imread(arguments['image'])
    pre_image, ratio = preprocess_image(image)

    # find edges
    edges = auto_canny(pre_image)

    cv2.imwrite('edges.jpg', edges) # debug

    # find contours of the page
    screen_contours = process_contours(edges)

    # transform the image
    warped_image = four_point_transform(image,
        screen_contours.reshape(4, 2) * ratio)

    # preprocess image for landmark detection
    landmark_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    landmark_image = cv2.GaussianBlur(landmark_image, (7, 7), 0)

    # detect edges, remove gaps between edges
    landmark_edges = auto_canny(landmark_image)
    landmark_edges = cv2.dilate(landmark_edges, None, iterations = 5)
    landmark_edges = cv2.erode(landmark_edges, None, iterations = 5)

    cv2.imwrite('output.jpg', landmark_edges) # debug

    # find, sort contours
    mark_contours = cv2.findContours(landmark_edges.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    
    mark_contours = mark_contours[1]

    (mark_contours, _) = contours.sort_contours(mark_contours)

    # loop over contours 
    for contour in mark_contours:
        if cv2.contourArea(contour) < 100: # tweak this value
            continue

        # finds rectangle covering the landmark
        bounding_box = cv2.minAreaRect(contour)

        # remove artifact straight lines
        (points, dimension, angle) = bounding_box
        if dimension[0] or dimension[1] < 7: # tweak this value
            if points[0] or points[1] < 3: # tweak this value
                if abs(angle) < 5: # tweak this value
                    continue

        # points for bounding box from rectangle
        bounding_box = cv2.boxPoints(bounding_box)
        bounding_box = np.array(bounding_box, dtype='int')
        bounding_box = perspective.order_points(bounding_box)
        tl = bounding_box[0]

        # distance between top-left corner and landmark
        distance_list.append(int(dist.euclidean((0, 0), (tl[0], tl[1]))))

    print('points   :', distance_list) # debug
    print('count    :', len(distance_list)) # debug
        

# call the main function
main()
