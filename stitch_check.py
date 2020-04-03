import cv2
import numpy as np
import utility as utl
import ransac
from image import Image
from stitch import stitch

if __name__ == '__main__':
    ransac_match_images = ['project_images/Rainier1.png', 'project_images/Rainier2.png']
    images = []
    print("Detecting SIFT keypoints and corresponding descriptors in the images.")
    for argumentImage in ransac_match_images:
        image = Image(argumentImage)
        image.detect_harris_corners()
        # image.detect_sift_corners()
        images.append(image)

    print("Calculating the matches between the argument images.")
    matches, matching_image = utl.get_matches(images[0], images[1])

    print("Finding inlier matches")
    homography, inverse_homography, match_image, number_of_inliers = ransac.RANSAC(matches=matches, numMatches=4, numIterations=150, inlierThreshold=50, hom=None, homInv=None, image1Display=images[0], image2Display=images[1])

    print("Stitching the image")
    stitched_image = stitch(images[0], images[1], homography, inverse_homography)
    cv2.imwrite('Results/project_images/4.png', stitched_image)