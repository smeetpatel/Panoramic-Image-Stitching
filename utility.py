import cv2
import numpy as np
from Corners import Corner

def get_gradient_x(image):
    # define sobel kernel
    sobel_x_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return cv2.filter2D(image, -1, sobel_x_kernel)

def get_gradient_y(image):
    # define sobel kernel
    sobel_y_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return cv2.filter2D(image, -1, sobel_y_kernel)

def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_orientation_bin(angle):
    if angle>=0 and angle<11.25:
        return 1
    elif angle>=11.25 and angle<22.5:
        return 2
    elif angle>=22.5 and angle<33.75:
        return 3
    elif angle>=33.75 and angle<45.00:
        return 4
    elif angle>=45.00 and angle<56.25:
        return 5
    elif angle>=56.25 and angle<67.50:
        return 6
    elif angle>=67.50 and angle<78.75:
        return 7
    elif angle>=78.75 and angle<=90.00:
        return 8

def get_normalized_vector(vector):
    if int(np.linalg.norm(vector))==0:
        return vector
    else:
        return [x/np.linalg.norm(vector) for x in vector]

def save_image(keypoints, image, title):
    for keypoint in keypoints:
        x = int(keypoint.pt[1])
        y = int(keypoint.pt[0])
        image[x][y] = (0, 0, 255)
    title = title.replace("image_sets", "Results")
    cv2.imwrite(title, image)

def get_matches(image_corners_1, image_corners_2):
    keypoints_1 = image_corners_1.keypoints_descriptors
    keypoints_2 = image_corners_2.keypoints_descriptors

    matches = []
    test = []

    threshold = 0.3
    kp = []
    for i in range(len(keypoints_1)):
        best_matches = []
        for j in range(len(keypoints_2)):
            dist = np.sum(np.square(np.subtract(keypoints_1[i], keypoints_2[j])))
            if dist<threshold:
                best_matches.append((dist, j))
            test.append(dist)
        if len(best_matches)>=2:
            best_matches = sorted(best_matches, key=lambda x:x[0])
            if best_matches[0][0]<0.80*best_matches[1][0]:
                kp.append(i)
                matches.append(cv2.DMatch(len(kp)-1, best_matches[0][1], best_matches[0][0]))
    return kp,matches