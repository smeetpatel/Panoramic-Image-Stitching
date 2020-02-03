import cv2
import numpy as np

def get_image_gradient_x(image):
    #define sobel kernel
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return cv2.filter2D(image, -1, sobel_x_kernel)

def get_image_gradient_y(image):
    #define sobel kernel
    sobel_y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
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
    image_path = "image_sets/graf/img1.ppm"
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
    count = 0
    for i in range(1, s_x_2.shape[0]-1):
        for j in range(1, s_x_2.shape[1]-1):
            harris_matrix = np.zeros((2,2), np.float32)
            for k in range(-1,2):
                for l in range(-1,2):
                    harris_matrix[0][0] += s_x_2[i + k][j + l]
                    harris_matrix[1][1] += s_y_2[i + k][j + l]
                    harris_matrix[0][1] += s_x_y[i + k][j + l]
            harris_matrix[1][0] = harris_matrix[0][1]
            corner_responses[i-1][j-1] = np.linalg.det(harris_matrix)/np.trace(harris_matrix)

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
    print("count: ", count)






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