import cv2
import numpy as np
import utility as utl
import ransac
from image import Image
from stitch import stitch

"""
This module demonstrates the following:

1) Implementation of stitching algorithm to stitch multiple images captured by me:
    - Calculate SIFT corners and descriptors, find matches, filter out best matches and calculate apt homography
    - For multiple images:
        - Select first image
        - Select image to stitch it with based on maximum number of inlier matches with the argument image
        - Stitch the selected image
        - Detect the corners for this new stitched image
        - Repeat this process to find next image to stitch and so on till all images are stitched.
"""
if __name__ == '__main__':
    ransac_match_images = images = ['misc/personal_1.jpg', 'misc/personal_2.jpg', 'misc/personal_3.jpg']
    images = []
    print("Detecting SIFT keypoints and corresponding descriptors in the images.")
    for argumentImage in ransac_match_images:
        image = Image(argumentImage)
        image.detect_sift_corners()
        images.append(image)

    # stitch images
    final_image = images[0]
    images = images[1:]
    index = 1
    while True:
        print("Adding image ", index)
        index += 1
        max_inliers = 0
        max_inliers_index = 0
        max_inliers_homography = None
        max_inliers_inverse_homography = None
        for i in range(len(images)):
            matches, matching_image = utl.get_matches(final_image, images[i])
            homography, inverse_homography, match_image, number_of_inliers = ransac.RANSAC(matches=matches, numMatches=4, numIterations=150, inlierThreshold=50, hom=None, homInv=None, image1Display=final_image, image2Display=images[i])
            if number_of_inliers > max_inliers:
                max_inliers = number_of_inliers
                max_inliers_index = i
                max_inliers_homography = homography
                max_inliers_inverse_homography = inverse_homography
        print("Best current match: ", images[max_inliers_index].image_name)
        print("Stitching")
        final_image = stitch(final_image, images[max_inliers_index], max_inliers_homography, max_inliers_inverse_homography)
        images = images[:max_inliers_index] + images[max_inliers_index + 1:]
        if len(images) == 0:
            break
        print("Finding keypoints in the stitched image")
        cv2.imwrite("image_sets/misc/mid.png", final_image)
        final_image = Image("misc/mid.png")
        final_image.detect_sift_corners()

    cv2.imwrite('Results/project_images/personal_images_stitched.png', final_image)