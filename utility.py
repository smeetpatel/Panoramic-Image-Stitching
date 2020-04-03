import cv2
import math
import numpy as np
from shapely.geometry import Point, Polygon



# show/save image helper functions
def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(corners, image, title):
    keypoints = [corner.keypoint for corner in corners]
    for keypoint in keypoints:
        x = int(keypoint.pt[1])
        y = int(keypoint.pt[0])
        image[x][y] = (0, 0, 255)
    title = title.replace("image_sets", "Results")
    cv2.imwrite(title, image)


# Harris corner detector helper functions
def get_gradient_x(image):
    # define sobel kernel
    sobel_x_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return cv2.filter2D(image, -1, sobel_x_kernel)


def get_gradient_y(image):
    # define sobel kernel
    sobel_y_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return cv2.filter2D(image, -1, sobel_y_kernel)


def get_orientation_bin(angle):
    if angle >= 0 and angle < 11.25:
        return 1
    elif angle >= 11.25 and angle < 22.5:
        return 2
    elif angle >= 22.5 and angle < 33.75:
        return 3
    elif angle >= 33.75 and angle < 45.00:
        return 4
    elif angle >= 45.00 and angle < 56.25:
        return 5
    elif angle >= 56.25 and angle < 67.50:
        return 6
    elif angle >= 67.50 and angle < 78.75:
        return 7
    elif angle >= 78.75 and angle <= 90.00:
        return 8


def get_normalized_vector(vector):
    if int(np.linalg.norm(vector)) == 0:
        return vector
    else:
        return [x / np.linalg.norm(vector) for x in vector]


def result_image_name(image_name):
    if (image_name == "project_images/Boxes.png"):
        image_name = "project_images/1a.png"
    elif (image_name == "project_images/Rainier1.png"):
        image_name = "project_images/1b.png"
    elif (image_name == "project_images/Rainier2.png"):
        image_name = "project_images/1c.png"
    return "Results/" + image_name


# helper functions to get matches between images based on detected key points
def get_matches(image_1, image_2):
    image_1_keypoints = [corner.keypoint for corner in image_1.corners]
    image_1_descriptors = [corner.descriptor for corner in image_1.corners]

    image_2_keypoints = [corner.keypoint for corner in image_2.corners]
    image_2_descriptors = [corner.descriptor for corner in image_2.corners]

    matches = []
    threshold = 2300

    for i in range(len(image_1_keypoints)):
        best_matches = []
        for j in range(len(image_2_keypoints)):
            dist = np.sum(np.absolute(np.array(image_1_descriptors[i]) - np.array(image_2_descriptors[j])))
            if dist < threshold:
                best_matches.append((dist, j))
        if len(best_matches) >= 2:
            best_matches = sorted(best_matches, key=lambda x: x[0])
            if best_matches[0][0] < 0.75 * best_matches[1][0]:
                matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=best_matches[0][1], _distance=best_matches[0][0]))

    match_image = cv2.drawMatches(image_1.image, image_1_keypoints, image_2.image, image_2_keypoints,
                                  matches1to2=matches, outImg=np.array([]),
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matches, match_image


def match(image_1, image_2):
    image_1_keypoints = [corner.keypoint for corner in image_1.corners]
    image_1_descriptors = [corner.descriptor for corner in image_1.corners]

    image_2_keypoints = [corner.keypoint for corner in image_2.corners]
    image_2_descriptors = [corner.descriptor for corner in image_2.corners]
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(np.asarray(image_1_descriptors), np.asarray(image_2_descriptors), k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(image_1.image, image_1_keypoints, image_2.image, image_2_keypoints, good, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img3


# helper functions for RANSAC
def get_distance(actual_x, actual_y, projected_x, projected_y):
    return abs(actual_x - projected_x) + abs(actual_y - projected_y)


# helper functions for stitching the images
def get_empty_stitched_image(image_1, image_2_tl, image_2_tr, image_2_bl, image_2_br):
    max_x = max(len(image_1.image), image_2_tl[1], image_2_tr[1], image_2_bl[1], image_2_br[1])
    min_x = min(len(image_1.image), image_2_tl[1], image_2_tr[1], image_2_bl[1], image_2_br[1])
    max_y = max(len(image_1.image[0]), image_2_tl[0], image_2_tr[0], image_2_bl[0], image_2_br[0])
    min_y = min(len(image_1.image[0]), image_2_tl[0], image_2_tr[0], image_2_bl[0], image_2_br[0])

    if min_x < 0:
        stitched_image_height = max_x + abs(min_x)
    else:
        stitched_image_height = max_x
    if min_y < 0:
        stitched_image_width = max_y + abs(min_y)
    else:
        stitched_image_width = max_y
    # stitched_image = np.zeros((math.ceil(stitched_image_height), math.ceil(stitched_image_width), 3))
    stitched_image = np.zeros((math.ceil(stitched_image_height), math.ceil(stitched_image_width), 3))
    return stitched_image

def within_boundary_check(projected_point, image_2):
    image_2_boundary = np.array([[0, 0], [len(image_2.image[0]), 0], [len(image_2.image[0]), len(image_2.image)], [0, len(image_2.image)]])
    if cv2.pointPolygonTest(image_2_boundary, projected_point, False) == 1.0:
        return True
    return False