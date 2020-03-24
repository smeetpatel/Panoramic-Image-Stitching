import cv2
import numpy as np
import utility as utl
import ransac
from image import Image

if __name__ == '__main__':
    ransac_match_images = ['project_images/Rainier1.png', 'project_images/Rainier2.png']
    images = []
    print("Detecting SIFT keypoints and corresponding descriptors in the images.")
    for argumentImage in ransac_match_images:
        image = Image(argumentImage)
        image.detect_harris_corners()
        images.append(image)

    print("Calculating the matches between the argument images.")
    matches, matching_image = utl.get_matches(images[0], images[1])
    cv2.imwrite("Results/project_images/2.png", matching_image)

    print("Finding inlier matches")
    homography, inverse_homography, match_image = ransac.RANSAC(matches=matches, numMatches=4, numIterations=150, inlierThreshold=50, hom=None, homInv=None, image1Display=images[0], image2Display=images[1])
    # utl.show_image('Match image', match_image)
    cv2.imwrite("Results/project_images/3.png", match_image)