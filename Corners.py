import cv2
import numpy as np
import utility as utl
from math import sqrt

class Corner:

    def __init__(self):
        self.keypoints = []
        self.keypoints_descriptors = []

    def detect_harris_corners(self, image_path):
        image = cv2.imread(image_path)

        # convert to grayscale image
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)

        # calculate gradients
        gradient_x = utl.get_gradient_x(gray_img)
        gradient_y = utl.get_gradient_y(gray_img)

        # get gradient images required for Harris matrix
        gradient_x2 = np.multiply(gradient_x, gradient_x)
        gradient_y2 = np.multiply(gradient_y, gradient_y)
        gradient_xy = np.multiply(gradient_x, gradient_y)

        # apply Gaussian kernel to the gradient images required for Harris matrix
        gradient_x2 = cv2.GaussianBlur(gradient_x2, (5, 5), 0, borderType=cv2.BORDER_CONSTANT)
        gradient_y2 = cv2.GaussianBlur(gradient_y2, (5, 5), 0, borderType=cv2.BORDER_CONSTANT)
        gradient_xy = cv2.GaussianBlur(gradient_xy, (5, 5), 0, borderType=cv2.BORDER_CONSTANT)

        # calculate summation of gradient values around a window of 7x7
        print("Calculating Harris matrix around a window for each pixel")
        summation_kernel = np.ones((5, 5), np.float32)
        gradient_x2_summation = cv2.filter2D(gradient_x2, -1, summation_kernel, borderType=cv2.BORDER_CONSTANT)
        gradient_y2_summation = cv2.filter2D(gradient_y2, -1, summation_kernel, borderType=cv2.BORDER_CONSTANT)
        gradient_xy_summation = cv2.filter2D(gradient_xy, -1, summation_kernel, borderType=cv2.BORDER_CONSTANT)

        # calculate corner response for each pixel
        print("Calculating corner response for each pixel")
        corner_responses = np.zeros(gradient_x2.shape, np.float32)
        harris_matrix = np.zeros((2, 2), np.float32)
        for i in range(corner_responses.shape[0]):
            for j in range(corner_responses.shape[1]):
                harris_matrix = np.array([[gradient_x2_summation[i][j], gradient_xy_summation[i][j]], [gradient_xy_summation[i][j], gradient_y2_summation[i][j]]])
                corner_responses[i][j] = np.linalg.det(harris_matrix) / np.trace(harris_matrix)

        # threshold the corner respones
        threshold = 0.01 * np.amax(corner_responses).tolist()
        corner_responses[corner_responses < threshold] = 0

        # calculate image's gradient orientation and magnitude for non-maximum supression
        image_gradient_orientation = cv2.phase(gradient_x2_summation, gradient_y2_summation, angleInDegrees=True)
        image_gradient_magnitude = np.sqrt(np.add(np.multiply(gradient_x2_summation, gradient_x2_summation),
                                                  np.multiply(gradient_y2_summation, gradient_y2_summation)))

        # perform non-maximum supression
        print("Performing non-maximum supression")
        # image_gradient_orientation = np.pad(image_gradient_orientation, (3, 3), 'constant')
        corner_responses = np.pad(corner_responses, (3, 3), 'constant')
        # threshold = 10000
        stop_looping = False
        for i in range(3, corner_responses.shape[0] - 3):
            for j in range(3, corner_responses.shape[1] - 3):
                if corner_responses[i][j] > 0:
                    local_maxima = corner_responses[i][j]
                    # orientation_bin = utl.get_orientation_bin(image_gradient_orientation[i][j])
                    for k in range(-2, 3):
                        for l in range(-2, 3):
                            # if utl.get_orientation_bin(image_gradient_orientation[i+k][j+l])==orientation_bin:
                            if i + k == 0 and j + l == 0:
                                continue
                            elif corner_responses[i + k][j + l] > local_maxima:
                                corner_responses[i][j] = 0
                                stop_looping = True
                                break
                        if stop_looping:
                            break
                    if stop_looping:
                        stop_looping = False

        #perform adaptive non-maximum supression
        print("Performing adaptive non-maximum supression")
        corner_responses = corner_responses[3:corner_responses.shape[0] - 3, 3:corner_responses.shape[1] - 3]
        number_of_keypoints = 500
        robust_factor = 1.1
        self.perform_adaptive_non_maximum_supression(corner_responses, number_of_keypoints, robust_factor)

        # display keypoints
        print("Number of keypoints: ", len(self.keypoints))
        #corners_image = cv2.drawKeypoints(image, self.keypoints, outImage=None, color=(0, 0, 255), flags=0)
        utl.save_image(self.keypoints, image, image_path)

        # create sift descriptors for all corner points
        print("Creating SIFT descriptor for all the keypoints")
        image_gradient_orientation = np.pad(image_gradient_orientation, (8, 8), 'constant')
        image_gradient_magnitude = np.pad(image_gradient_magnitude, (8, 8), 'constant')
        for keypoint in self.keypoints:
            x = int(keypoint.pt[1]) + 8
            y = int(keypoint.pt[0]) + 8
            window = 16
            orientation_sub_patches = []
            magnitude_sub_patches = []
            desc = []

            # create sub patches
            """for i in range(y - (window // 2), y + (window // 2), 4):
                for j in range(x - (window // 2), x + (window // 2), 4):
                    orientation_sub_patches.append(image_gradient_orientation[j:j + 4, i:i+4])
                    magnitude_sub_patches.append(image_gradient_magnitude[j:j + 4, i:i+4])"""
            for i in range(x - (window // 2), x + (window // 2), 4):
                for j in range(y - (window // 2), y + (window // 2), 4):
                    orientation_sub_patches.append(image_gradient_orientation[i:i + 4, j:j + 4])
                    magnitude_sub_patches.append(image_gradient_magnitude[i:i + 4, j:j + 4])

            # make the patches align with x-axis based on the dominant orientation to achieve rotation invariance
            for i in range(len(orientation_sub_patches)):
                hist, bins = np.histogram(orientation_sub_patches[i], bins=[0, 11.25, 22.50, 33.75, 45.00, 56.25, 67.50, 78.75, 90.00])
                values_bin_indices = np.digitize(orientation_sub_patches[i], bins=bins)
                ele = []
                for k in range(values_bin_indices.shape[0]):
                    for l in range(values_bin_indices.shape[1]):
                        if values_bin_indices[k][l] - 1 == list(hist).index(int(hist.max())):
                            ele.append(orientation_sub_patches[i][k][l])
                dominant_orientation = np.array(ele).mean()
                #dominant_orientation = image_gradient_orientation[int(keypoint.pt[1])][int(keypoint.pt[0])]
                orientation_sub_patches[i] = [x - dominant_orientation for x in orientation_sub_patches[i]]

            # create histogram for feature descriptor and threshold normalize it to make it contrast invariant
            for i in range(len(orientation_sub_patches)):
                hist, bins = np.histogram(np.array(orientation_sub_patches[i]), bins=[-90.00, -67.5, -45.00, -22.50, 0.00, 22.50, 45.00, 67.50, 90.00], weights=np.array(magnitude_sub_patches[i]))
                hist = utl.get_normalized_vector(hist)
                for j in range(len(hist)):
                    if hist[j] > 0.2:
                        hist[j] = 0.2
                hist = utl.get_normalized_vector(hist)
                for j in hist:
                    desc.append(j)

            # store the desc as keypoint descriptor
            self.keypoints_descriptors.append(desc)
        print("SIFT descriptors are calculated.\n\n\n")

    def set_keypoints(self, keypoints):
        self.keypoints = keypoints

    def set_keydescriptors(self, keypoints_descriptor):
        self.keypoints_descriptors = keypoints_descriptor

    def perform_adaptive_non_maximum_supression(self, corner_responses, number_of_keypoints, robust_factor):
        temp_list = []
        for i in range(corner_responses.shape[0]):
            for j in range(corner_responses.shape[1]):
                if corner_responses[i][j] > 0:
                    temp_list.append(((i, j), corner_responses[i][j]))
        temp_list = sorted(temp_list, key=lambda x: x[1], reverse=True)
        radii = []
        for i in range(2, len(temp_list)):
            temp_radii = []
            for j in range(len(temp_list[:i])):
                if temp_list[i][1] < robust_factor * temp_list[j][1]:
                    temp_radii.append(sqrt((temp_list[i][0][0] - temp_list[j][0][0]) ** 2 + (
                                temp_list[i][0][1] - temp_list[j][0][1]) ** 2))
            temp_radii.sort()
            radii.append((temp_radii[0], i))
        radii = sorted(radii, key=lambda x: x[0], reverse=True)
        for i in range(len(radii[:number_of_keypoints])):
            point = temp_list[radii[i][1]][0]
            cr = temp_list[radii[i][1]][1]
            keypoint = cv2.KeyPoint(point[1], point[0], 0, _response=cr)
            self.keypoints.append(keypoint)