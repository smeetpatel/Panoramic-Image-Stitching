import cv2
import utility as utl
import numpy as np
from math import sqrt

class Image:

    def __init__(self, image_name):
        image_path = 'image_sets/' + image_name
        self.image_name = image_name
        self.image = cv2.imread(image_path)
        self.corners = []

    def detect_harris_corners(self):
        # convert image to grayscale image
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
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

        # define window size and offset for calculating corners and related further processing
        window = (7,7)
        offset = (-(int(window/2)), int(window/2)+1)

        # calculate summation of gradient values around a window of 7x7
        print("Calculating Harris matrix around a window for each pixel")
        summation_kernel = np.ones(window, np.float32)
        gradient_x2_summation = cv2.filter2D(gradient_x2, -1, summation_kernel, borderType=cv2.BORDER_CONSTANT)
        gradient_y2_summation = cv2.filter2D(gradient_y2, -1, summation_kernel, borderType=cv2.BORDER_CONSTANT)
        gradient_xy_summation = cv2.filter2D(gradient_xy, -1, summation_kernel, borderType=cv2.BORDER_CONSTANT)

        # calculate corner response for each pixel
        print("Calculating corner response for each pixel")
        corner_responses = np.zeros(gradient_x2.shape, np.float32)
        for i in range(corner_responses.shape[0]):
            for j in range(corner_responses.shape[1]):
                harris_matrix = np.array([[gradient_x2_summation[i][j], gradient_xy_summation[i][j]],
                                          [gradient_xy_summation[i][j], gradient_y2_summation[i][j]]])
                corner_responses[i][j] = np.linalg.det(harris_matrix) / np.trace(harris_matrix)

        # threshold the corner respones
        threshold = 0.01 * np.amax(corner_responses).tolist()
        corner_responses[corner_responses < threshold] = 0

        # perform non-maximum supression
        print("Performing non-maximum supression")
        corner_responses = np.pad(corner_responses, (abs(offset[0]), abs(offset[0])), 'constant')
        not_local_maxima = False
        for i in range(abs(offset[0]), corner_responses.shape[0] - abs(offset[0])):
            for j in range(abs(offset[0]), corner_responses.shape[1] - abs(offset[0])):
                if corner_responses[i][j] > 0:
                    local_maxima = corner_responses[i][j]
                    for k in range(offset[0], offset[1]):
                        for l in range(offset[0], offset[1]):
                            if i + k == 0 and j + l == 0:
                                continue
                            elif corner_responses[i + k][j + l] > local_maxima:
                                corner_responses[i][j] = 0
                                not_local_maxima = True
                                break
                        if not_local_maxima:
                            break
                    if not_local_maxima:
                        not_local_maxima = False

        # perform adaptive non-maximum supression
        print("Performing adaptive non-maximum supression")
        corner_responses = corner_responses[abs(offset[0]):corner_responses.shape[0] - abs(offset[0]), abs(offset[0]):corner_responses.shape[1] - abs(offset[0])]
        number_of_keypoints = 500
        robust_factor = 1.1
        self.perform_adaptive_non_maximum_supression(corner_responses, number_of_keypoints, robust_factor)

        # display keypoints
        print("Number of keypoints: ", len(self.corners))
        result_image_path = "Results/" + self.image_name
        utl.save_image(self.corners, self.image, result_image_path)

        # calculate SIFT descriptor for all the keypoints
        print("Creating SIFT descriptor for all the keypoints")
        image_gradient_orientation = cv2.phase(gradient_x2_summation, gradient_y2_summation, angleInDegrees=True)
        image_gradient_magnitude = np.sqrt(np.add(np.multiply(gradient_x2_summation, gradient_x2_summation),
                                                  np.multiply(gradient_y2_summation, gradient_y2_summation)))
        image_gradient_orientation = np.pad(image_gradient_orientation, (8, 8), 'constant')
        image_gradient_magnitude = np.pad(image_gradient_magnitude, (8, 8), 'constant')
        for corner in self.corners:
            x = int(corner.keypoint.pt[1]) + 8
            y = int(corner.keypoint.pt[0]) + 8
            window = 16
            orientation_sub_patches = []
            magnitude_sub_patches = []
            desc = []

            # create sub patches
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
            keypoint.set_descriptor(self, desc)
        print("SIFT descriptors are calculated.\n\n\n")

    def perform_adaptive_non_maximum_supression(self, corner_responses, number_of_keypoints, robust_factor):
        temp_list = []

        #make a list of tuples holding (coordinate tuple, response_value) and sort it based on the response value
        for i in range(corner_responses.shape[0]):
            for j in range(corner_responses.shape[1]):
                if corner_responses[i][j] > 0:
                    temp_list.append(((i, j), corner_responses[i][j]))
        temp_list = sorted(temp_list, key=lambda x: x[1], reverse=True)

        #Calculate radius of supression for all points and sort the list in ascending manner based on supression radii
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
            self.corners.append(Corner(keypoint))


class Corner:
    def __init__(self, keypoint):
        self.keypoint = keypoint
        self.descriptor = []

    def set_descriptor(self, desc):
        self.descriptor = desc

