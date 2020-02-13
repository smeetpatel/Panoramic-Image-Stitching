import cv2
import numpy as np

def get_image_gradient_x(image):
    #define sobel kernel
    sobel_x_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return cv2.filter2D(image, -1, sobel_x_kernel)

def get_image_gradient_y(image):
    #define sobel kernel
    sobel_y_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return cv2.filter2D(image, -1, sobel_y_kernel)

def get_harris_value(s_x_2, s_y_2, s_x_y):
    shape = s_x_2.shape
    img = np.zeros(shape, np.float32)
    #harris_matrix = np.zeros((2,2), np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            harris_matrix = np.array([[s_x_2[i][j], s_x_y[i][j]], [s_x_y[i][j], s_y_2[i][j]]])
            img[i][j] = np.float32(np.linalg.det(harris_matrix)/np.trace(harris_matrix))
    return img

if __name__ == '__main__':
    #read image
    image_path = "image_sets/yosemite/Yosemite1.jpg"
    image = cv2.imread(image_path)

    """orb = cv2.ORB_create()
    kp = orb.detect(image, None)
    sobel_img = cv2.Laplacian(image, cv2.CV_8U)
    kp, des = orb.compute(image, kp)
    img2 = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
    cv2.imshow('image2', img2)"""

    #convert to grayscale image
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    #calculate i_x, i_y, i_x^2, i_y^2, i_x*i_y
    i_x = get_image_gradient_x(gray_img)
    i_y = get_image_gradient_y(gray_img)
    i_x_2 = np.multiply(i_x, i_x)
    i_y_2 = np.multiply(i_y, i_y)
    i_x_i_y = np.multiply(i_x, i_y)

    #apply Gaussian to each gradient images
    s_x_2 = cv2.GaussianBlur(i_x_2, (3,3), 0)
    s_y_2 = cv2.GaussianBlur(i_y_2, (3,3), 0)
    s_x_y = cv2.GaussianBlur(i_x_i_y, (3,3), 0)

    #pad the gradient images to calculate Harris matrix
    s_x_2 = np.pad(s_x_2, ((1, 1), (1,1)), 'constant')
    s_y_2 = np.pad(s_y_2, ((1, 1), (1, 1)), 'constant')
    s_x_y = np.pad(s_x_y, ((1, 1), (1, 1)), 'constant')

    #calculate the corner responses for every pixel of the image
    harris_matrix = np.zeros((2,2), np.float32)
    corner_responses = np.zeros(s_x_2.shape, np.float32)
    for i in range(1, s_x_2.shape[0]-1):
        for j in range(1, s_x_2.shape[1]-1):
            harris_matrix = np.zeros((2,2), np.float32)
            for k in range(-1,2):
                for l in range(-1,2):
                    harris_matrix[0][0] += s_x_2[i + k][j + l]
                    harris_matrix[1][1] += s_y_2[i + k][j + l]
                    harris_matrix[0][1] += s_x_y[i + k][j + l]
            harris_matrix[1][0] = harris_matrix[0][1]
            corner_responses[i][j] = np.linalg.det(harris_matrix)/np.trace(harris_matrix)

    #calculate the key points of interest
    threshold = 0.01*np.amax(corner_responses)
    keypoints = []
    for i in range(1, corner_responses.shape[0]-1):
        for j in range(1, corner_responses.shape[1]-1):
            if(corner_responses[i][j]>=threshold):
                max = True
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        if(corner_responses[i][j]<corner_responses[i+k][j+l]):
                            max = False;
                if max:
                    keypoints.append(cv2.KeyPoint(i, j, 1))
    img2 = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    cv2.imshow('image2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(len(keypoints))

    print("Corner responses minimum: ", np.amin(corner_responses))
    print("Corner responses maximum: ", np.amax(corner_responses))

    """harris_value = get_harris_value(s_x_2, s_y_2, s_x_y)
    print("Minimum: ", np.amin(harris_value))
    print("Maximum: ", np.amax(harris_value))"""

     #Corner detection using Harris Corner detection method implemented in OpenCV
    dst = cv2.cornerHarris(gray_img, 2, 3, 0.04)
    print(dst.shape)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]

    #show corners detected by Harris corner detection method implemented in OpenCV
    cv2.imshow("Corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """#perform non-maximum supression
        image_gradient_orientation = np.pad(image_gradient_orientation, (3,3), 'constant')
        image_gradient_magnitude = np.pad(image_gradient_magnitude, (3,3), 'constant')
        threshold = 0.01*np.amax(corner_responses).tolist()
        #threshold = 10000
        keypoints = []
        stop_looping = False
        """"""new_img = image.copy()
        new_img[corner_responses>threshold] = [0,0,255]
        utl.show_image("Corners", new_img)""""""
        """"""for i in range(corner_responses.shape[0]):
            for j in range(corner_responses.shape[1]):
                if corner_responses[i][j] >= threshold:
                    keypoints.append(cv2.KeyPoint(i, j, 0))""""""

        for i in range(3, image_gradient_magnitude.shape[0]-3):
            for j in range(3, image_gradient_magnitude.shape[1]-3):

                #process only the points whose corner response is greater than the threshold
                if corner_responses[i-3][j-3] >= threshold:
                    local_maxima = corner_responses[i-3][j-3]
                    orientation_bin = utl.get_orientation_bin(image_gradient_orientation[i][j])

                    for k in range(-3,4):
                        for l in range(-3,4):

                            #process only the points whose orientation is similar to the orientation of the point in consideration
                            if utl.get_orientation_bin(image_gradient_orientation[i+k][j+l])==orientation_bin:
                                if i+k==0 and j+l==0:
                                    continue
                                elif image_gradient_magnitude[i+k][j+l]>=local_maxima:
                                    corner_responses[i-3][j-3] = 0
                                    image_gradient_magnitude[]
                                    stop_looping = True
                                    break
                                else:
                                    image_gradient_magnitude[i + k][j + l] = 0
                        if stop_looping:
                            break
                    if stop_looping:
                        stop_looping = False
                else:
                    #corner_responses[i - 3][j - 3] = 0
                    continue

        keypoints = np.argwhere(image_gradient_magnitude > 0)
        keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints]
        print("Number of keypoints: ", len(keypoints))
        corners_image = cv2.drawKeypoints(image, keypoints, outImage=None, color=(0,255,0), flags=0)
        utl.show_image("Keypoints in the image", corners_image)"""

    """cv2.imwrite("X_gradient.jpg", gradient_x)
    cv2.imwrite("Y_gradient.jpg", gradient_y)

    cv2.imwrite("X2_gradient.jpg", gradient_x2)
    cv2.imwrite("Y2_gradient.jpg", gradient_y2)
    cv2.imwrite("XY_gradient.jpg", gradient_xy)

    cv2.imwrite("Gaussian X2 gradient.jpg", gradient_x2)
    cv2.imwrite("Gaussian Y2 gradient.jpg", gradient_y2)
    cv2.imwrite("Gaussian XY gradient.jpg", gradient_xy)

    # get max and min of corner_response
    print("Maximum: ", np.amax(corner_responses))
    print("Minimum: ", np.amin(corner_responses))
    print("Mean: ", np.mean(corner_responses))
    max = 0.01 * np.amax(corner_responses).tolist()
    count = 0
    for i in np.nditer(corner_responses):
        if i >= max:
            count += 1
    print("count: ", count)

    print("Gradient angle maximum: ", np.amax(image_gradient_orientation))
    print("Gradient angle minimum: ", np.amin(image_gradient_orientation))"""