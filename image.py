import cv2
import utility as utl
import numpy as np

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
        gradient_x2 = cv2.GaussianBlur(gradient_x2, (3, 3), 0, borderType=cv2.BORDER_CONSTANT)
        gradient_y2 = cv2.GaussianBlur(gradient_y2, (3, 3), 0, borderType=cv2.BORDER_CONSTANT)
        gradient_xy = cv2.GaussianBlur(gradient_xy, (3, 3), 0, borderType=cv2.BORDER_CONSTANT)

        # define window size and offset for calculating corners and related further processing
        window = (3, 3)
        offset = (-(int(window[0] / 2)), int(window[0] / 2) + 1)

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
                if np.trace(harris_matrix) == 0:
                    corner_responses[i][j] = 0
                else:
                    corner_responses[i][j] = np.linalg.det(harris_matrix) / np.trace(harris_matrix)

        #scale corner responses between 0 and 255
        dst = np.empty(corner_responses.shape, dtype=np.float32)
        cv2.normalize(corner_responses, dst=dst, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        corner_responses = cv2.convertScaleAbs(dst)

        # threshold the corner respones
        threshold = 150
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
                    else:
                        keypoint = cv2.KeyPoint(j-abs(offset[0]), i-abs(offset[0]), 0)
                        self.corners.append(Corner(keypoint))

        # display number of corners detected and save the image showing detected corners
        number_of_NMS_keypoints = len(self.corners)
        print("Number of keypoints after NMS: ", number_of_NMS_keypoints)
        keypoints = []
        for corner in self.corners:
            keypoints.append(corner.keypoint)
        output = cv2.drawKeypoints(self.image, keypoints, np.array([]), color=(0, 0, 255))
        result_image_path = utl.result_image_name(self.image_name)
        cv2.imwrite(result_image_path, img=output)

        # # perform adaptive non-maximum supression
        # self.perform_adaptive_non_maximum_supression(corner_responses, number_of_NMS_keypoints, offset)

        # calculate SIFT descriptor for all the keypoints
        self.compute_custom_sift_descriptor(gradient_x2_summation, gradient_y2_summation)
        # self.compute_opencv_sift_descriptor()

    def detect_sift_corners(self):
        # convert image to grayscale image
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        #compute keypoints and descriptors
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray_img, None)

        #save images with the detected keypoints
        # img = cv2.drawKeypoints(gray_img, kp, self.image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img = cv2.drawKeypoints(gray_img, kp, self.image)
        result_image_path = utl.result_image_name(self.image_name)
        cv2.imwrite(result_image_path, img=img)

        #register the detected keypoints and corners in the list self.corners
        for i in range(len(kp)):
            self.corners.append(Corner(kp[i]))
            self.corners[i].set_descriptor(des[i])

    def perform_adaptive_non_maximum_supression(self, corner_responses, number_of_NMS_keypoints, offset):
        print("Performing adaptive non-maximum supression")
        corner_responses = corner_responses[abs(offset[0]):corner_responses.shape[0] - abs(offset[0]),
                           abs(offset[0]):corner_responses.shape[1] - abs(offset[0])]
        if(number_of_NMS_keypoints>500):
            number_of_keypoints = 500
        else:
            number_of_keypoints = number_of_NMS_keypoints
        # self.adaptive_non_maximum_supression(number_of_keypoints)

        # # number_of_keypoints = 500
        robust_factor = 1.1
        self.perform_ANMS(corner_responses, number_of_keypoints, robust_factor)

        # show corner response
        utl.show_image('Corner response after ANMS', corner_responses)

        # display keypoints
        print("Number of keypoints: ", len(self.corners))
        result_image_path = utl.result_image_name(self.image_name)
        utl.save_image(self.corners, self.image, result_image_path)

    def perform_ANMS(self, corner_responses, number_of_keypoints, robust_factor):
        self.corners = []
        temp_list = []

        # make a list of tuples holding (coordinate tuple, response_value) and sort it based on the response value
        for i in range(corner_responses.shape[0]):
            for j in range(corner_responses.shape[1]):
                if corner_responses[i][j] > 0:
                    temp_list.append(((i, j), corner_responses[i][j]))
        temp_list = sorted(temp_list, key=lambda x: x[1], reverse=True)

        # Calculate radius of supression for all points and sort the list in ascending manner based on supression radii
        radii = []
        for i in range(2, len(temp_list)):
            temp_radii = []
            for j in range(len(temp_list[:i])):
                if temp_list[i][1] <= robust_factor * temp_list[j][1]:
                    temp_radii.append(sqrt((temp_list[i][0][0] - temp_list[j][0][0]) ** 2 + (
                            temp_list[i][0][1] - temp_list[j][0][1]) ** 2))
            print(temp_list[i][0][0], "\t", temp_list[i][0][1])
            temp_radii.sort()
            radii.append((temp_radii[0], i))
        radii = sorted(radii, key=lambda x: x[0], reverse=True)

        for i in range(len(radii[:number_of_keypoints])):
            point = temp_list[radii[i][1]][0]
            cr = temp_list[radii[i][1]][1]
            keypoint = cv2.KeyPoint(point[1], point[0], 0, _response=cr)
            self.corners.append(Corner(keypoint))

    def compute_custom_sift_descriptor(self, gradient_x2_summation, gradient_y2_summation):
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
                hist, bins = np.histogram(orientation_sub_patches[i],
                                          bins=[0, 11.25, 22.50, 33.75, 45.00, 56.25, 67.50, 78.75, 90.00])
                values_bin_indices = np.digitize(orientation_sub_patches[i], bins=bins)
                ele = []
                for k in range(values_bin_indices.shape[0]):
                    for l in range(values_bin_indices.shape[1]):
                        if values_bin_indices[k][l] - 1 == list(hist).index(int(hist.max())):
                            ele.append(orientation_sub_patches[i][k][l])
                dominant_orientation = np.array(ele).mean()
                # dominant_orientation = image_gradient_orientation[int(keypoint.pt[1])][int(keypoint.pt[0])]
                orientation_sub_patches[i] = [x - dominant_orientation for x in orientation_sub_patches[i]]

            # create histogram for feature descriptor and threshold normalize it to make it contrast invariant
            for i in range(len(orientation_sub_patches)):
                hist, bins = np.histogram(np.array(orientation_sub_patches[i]),
                                          bins=[-90.00, -67.5, -45.00, -22.50, 0.00, 22.50, 45.00, 67.50, 90.00],
                                          weights=np.array(magnitude_sub_patches[i]))
                hist = utl.get_normalized_vector(hist)
                for j in range(len(hist)):
                    if hist[j] > 0.2:
                        hist[j] = 0.2
                hist = utl.get_normalized_vector(hist)
                for j in hist:
                    desc.append(j)

            # store the desc as keypoint descriptor
            corner.set_descriptor(desc)
        print("SIFT descriptors are calculated.\n\n\n")

class Corner:
    def __init__(self, keypoint):
        self.keypoint = keypoint
        self.descriptor = []

    def set_descriptor(self, desc):
        self.descriptor = desc