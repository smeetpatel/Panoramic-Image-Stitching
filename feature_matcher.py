import cv2
import numpy as np
import utility as utl
from Corners import Corner

if __name__ == '__main__':
    user_input = input()



    #get image path
    #image_path = "image_sets/yosemite/Yosemite1.jpg"
    image_path = "image_sets/graf/img1.ppm"
    #image_path = "image_sets/panorama/pano1_0008.png"
    print("Calculate corners for: ", image_path)
    img1 = cv2.imread(image_path)
    image_corners_1 = Corner()
    image_corners_1.detect_harris_corners(image_path)

    #image_path = "image_sets/yosemite/Yosemite2.jpg"
    image_path = "image_sets/graf/img2.ppm"
    #image_path = "image_sets/panorama/pano1_0009.png"
    print("Calculate corners for: ", image_path)
    img2 = cv2.imread(image_path)
    image_corners_2 = Corner()
    image_corners_2.detect_harris_corners(image_path)

    """image_corners_1 = Corner()
    image_corners_2 = Corner()
    image_corners_1.set_keydescriptors(np.load("kd1.npy"))
    image_corners_2.set_keydescriptors(np.load("kd2.npy"))"""

    """#image_path = "image_sets/graf/img4.ppm"
    image_path = "image_sets/panorama/pano1_0010.png"
    print("Calculate corners for: ", image_path)
    image_corners_3 = Corner()
    image_corners_3.detect_harris_corners(image_path)

    image_path = "image_sets/panorama/pano1_0011.png"
    print("Calculate corners for: ", image_path)
    image_corners_4 = Corner()
    image_corners_4.detect_harris_corners(image_path)"""


    kp, matches = utl.get_matches(image_corners_1, image_corners_2)
    print("Matches found: ", len(matches))
    keypoints_1 = []
    """for i in range(len(image_corners_1.keypoints)):
        if i in kp:
            keypoints_1.append(image_corners_1.keypoints[i])"""
    for i in kp:
       keypoints_1.append(image_corners_1.keypoints[i])
    match = cv2.drawMatches(img1, keypoints_1, img2, image_corners_2.keypoints, matches1to2=matches[:10], outImg=None, flags=2)
    utl.show_image("Matches", match)
    cv2.imwrite("Match.jpg", match)