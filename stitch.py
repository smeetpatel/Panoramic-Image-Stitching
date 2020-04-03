import cv2
import math
import numpy as np
import utility as utl
from ransac import project

def stitch(image_1, image_2, homography, inverse_homography):
    # project all four corners of image_2 on image_1
    image_2_tl = project(0, 0, inverse_homography)
    image_2_tr = project(len(image_2.image[0]), 0, inverse_homography)
    image_2_bl = project(0, len(image_2.image), inverse_homography)
    image_2_br = project(len(image_2.image[0]), len(image_2.image), inverse_homography)

    # create an array to hold stitched image
    stitched_image = utl.get_empty_stitched_image(image_1, image_2_tl, image_2_tr, image_2_bl, image_2_br)

    # copy image 1 at right place in the stitched_image array
    if min(image_2_tl[1], image_2_tr[1]) < 0:
        leave_top = math.floor(abs(min(image_2_tl[1], image_2_tr[1])))
    else:
        leave_top = 0
    if min(image_2_tl[0], image_2_bl[0]) < 0:
        leave_left = math.floor(abs(min(image_2_tl[0], image_2_bl[0])))
    else:
        leave_left = 0
    img1_row = 0
    for row in range(leave_top, leave_top + len(image_1.image)):
        img1_col = 0
        for column in range(leave_left, leave_left + len(image_1.image[0])):
            projected_point = project(img1_col, img1_row, homography)
            # if utl.within_boundary_check(projected_point, image_2_tl, image_2_tr, image_2_bl, image_2_br):
            if utl.within_boundary_check(projected_point, image_2):
                dst = cv2.getRectSubPix(image=image_2.image, patchSize=(1,1), center=projected_point)
                stitched_image[row][column] = dst[0][0]
            else:
                stitched_image[row][column] = image_1.image[img1_row][img1_col]
            img1_col = img1_col + 1
        img1_row = img1_row + 1

    # project the points of stitched image on image_2 and get its value by bi-linear interpolation
    for row in range(len(stitched_image)):
        for column in range(len(stitched_image[0])):
            projected_point = project(column - leave_left, row - leave_top, homography)
            # if utl.within_boundary_check(projected_point, image_2_tl, image_2_tr, image_2_bl, image_2_br):
            if utl.within_boundary_check(projected_point, image_2):
                dst = cv2.getRectSubPix(image=image_2.image, patchSize=(1,1), center=projected_point)
                stitched_image[row][column] = dst[0][0]

    #return the stitched_image
    return stitched_image