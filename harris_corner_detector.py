import cv2
import numpy as np
import utility as utl

if __name__ == '__main__':
    # read image
    #image_path = "image_sets/yosemite/Yosemite1.jpg"
    image_path = "image_sets/graf/img1.ppm"
    image = cv2.imread(image_path)

    # convert to grayscale image
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    #calculate gradients
    gradient_x = utl.get_gradient_x(gray_img)
    gradient_y = utl.get_gradient_y(gray_img)

    #get gradient images required for Harris matrix
    gradient_x2 = np.multiply(gradient_x, gradient_x)
    gradient_y2 = np.multiply(gradient_y, gradient_y)
    gradient_xy = np.multiply(gradient_x, gradient_y)

    #apply Gaussian kernel to the gradient images required for Harris matrix
    gradient_x2 = cv2.GaussianBlur(gradient_x2, (5, 5), 0, borderType=cv2.BORDER_CONSTANT)
    gradient_y2 = cv2.GaussianBlur(gradient_y2, (5, 5), 0, borderType=cv2.BORDER_CONSTANT)
    gradient_xy = cv2.GaussianBlur(gradient_xy, (5, 5), 0, borderType=cv2.BORDER_CONSTANT)

    #calculate summation of gradient values around a window of 7x7
    summation_kernel = np.ones((5,5), np.float32)
    gradient_x2_summation = cv2.filter2D(gradient_x2, -1, summation_kernel, borderType=cv2.BORDER_CONSTANT)
    gradient_y2_summation = cv2.filter2D(gradient_y2, -1, summation_kernel, borderType=cv2.BORDER_CONSTANT)
    gradient_xy_summation = cv2.filter2D(gradient_xy, -1, summation_kernel, borderType=cv2.BORDER_CONSTANT)

    #calculate corner response for each pixel
    corner_responses = np.zeros(gradient_x2.shape, np.float32)
    harris_matrix = np.zeros((2,2), np.float32)
    for i in range(corner_responses.shape[0]):
        for j in range(corner_responses.shape[1]):
            #harris_matrix = [[gradient_x2_summation[i][j], gradient_xy_summation[i][j]], [gradient_xy_summation[i][j], gradient_y2_summation[i][j]]]
            harris_matrix[0][0] = gradient_x2_summation[i][j]
            harris_matrix[0][1] = gradient_xy_summation[i][j]
            harris_matrix[1][0] = gradient_xy_summation[i][j]
            harris_matrix[1][1] = gradient_y2_summation[i][j]
            corner_responses[i][j] = np.linalg.det(harris_matrix)/np.trace(harris_matrix)

    #threshold the corner respones
    threshold = 0.10 * np.amax(corner_responses).tolist()
    corner_responses[corner_responses < threshold] = 0

    #calculate image's gradient orientation and magnitude for non-maximum supression
    image_gradient_orientation = cv2.phase(gradient_x2_summation, gradient_y2_summation, angleInDegrees=True)
    image_gradient_magnitude = np.sqrt(np.add(np.multiply(gradient_x2_summation,gradient_x2_summation), np.multiply(gradient_y2_summation,gradient_y2_summation)))

    #perform non-maximum supression
    #image_gradient_orientation = np.pad(image_gradient_orientation, (3, 3), 'constant')
    corner_responses = np.pad(corner_responses, (3,3), 'constant')
    #threshold = 10000
    stop_looping = False
    for i in range(3, corner_responses.shape[0]-3):
        for j in range(3, corner_responses.shape[1] - 3):
            if corner_responses[i][j] > 0:
                local_maxima = corner_responses[i][j]
                #orientation_bin = utl.get_orientation_bin(image_gradient_orientation[i][j])
                for k in range(-2,3):
                    for l in range(-2,3):
                        #if utl.get_orientation_bin(image_gradient_orientation[i+k][j+l])==orientation_bin:
                        if i + k == 0 and j + l == 0:
                            continue
                        elif corner_responses[i + k][j + l]>local_maxima:
                            corner_responses[i][j] = 0
                            stop_looping = True
                            break
                    if stop_looping:
                        break
                if stop_looping:
                    stop_looping = False

    #fetch keypoints
    corner_responses = corner_responses[3:corner_responses.shape[0]-3, 3:corner_responses.shape[1]-3]
    keypoints = np.argwhere(corner_responses > 0)
    keypoints = [cv2.KeyPoint(x[1], x[0], 0) for x in keypoints]

    #display keypoints
    print("Number of keypoints: ", len(keypoints))
    corners_image = cv2.drawKeypoints(image, keypoints, outImage=None, color=(0, 0, 255), flags=0)
    utl.show_image("Keypoints in the image", corners_image)

    #create sift descriptors for all corner points
    image_gradient_orientation = np.pad(image_gradient_orientation, (8, 8), 'constant')
    image_gradient_magnitude = np.pad(image_gradient_magnitude, (8, 8), 'constant')
    keypoint_descriptors = []
    for keypoint in keypoints:
        x = int(keypoint.pt[1]) + 8
        y = int(keypoint.pt[0]) + 8
        window = 16
        orientation_sub_patches = []
        magnitude_sub_patches = []
        desc = []

        #create sub patches
        for i in range(y - (window // 2), y + (window // 2), 4):
            for j in range(x - (window // 2), x + (window // 2), 4):
                orientation_sub_patches.append(image_gradient_orientation[i:i + 4, j:j + 4])
                magnitude_sub_patches.append(image_gradient_magnitude[i:i + 4, j:j + 4])

        #make the patches align with x-axis based on the dominant orientation to achieve rotation invariance
        for i in range(len(orientation_sub_patches)):
            hist, bins  = np.histogram(orientation_sub_patches[i], bins=[0, 11.25, 22.50, 33.75, 45.00, 56.25, 67.50, 78.75, 90.00])
            values_bin_indices = np.digitize(orientation_sub_patches[i], bins=bins)
            ele = []
            for k in range(values_bin_indices.shape[0]):
                for l in range(values_bin_indices.shape[1]):
                    if values_bin_indices[k][l]-1==list(hist).index(int(hist.max())):
                        ele.append(orientation_sub_patches[i][k][l])
            dominant_orientation = np.array(ele).mean()
            orientation_sub_patches[i] = [x-dominant_orientation for x in orientation_sub_patches[i]]

        #create histogram
        for i in range(len(orientation_sub_patches)):
            hist, bins = np.histogram(orientation_sub_patches[i], bins=[-90.00, -67.5, -45.00, -22.50, 0, 22.50, 45.00, 67.50, 90.00], weights=magnitude_sub_patches[i])
            hist = [x/np.linalg.norm(hist) for x in hist]
            for j in range(len(hist)):
                if j>0.2:
                    hist[i] = 0.2
            hist = [x/np.linalg.norm(hist) for x in hist]
            for j in hist:
                desc.append(j)

        #store the desc as keypoint descriptor
        keypoint_descriptors.append(desc)