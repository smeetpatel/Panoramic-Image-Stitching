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

if __name__ == '__main__':
    #read image
    image_path = "image_sets/graf/img1.ppm"
    image = cv2.imread(image_path)

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
    i_x_2 = cv2.GaussianBlur(i_x_2, (3,3), 0)
    i_y_2 = cv2.GaussianBlur(i_y_2, (3,3), 0)
    i_x_i_y = cv2.GaussianBlur(i_x_i_y, (3,3), 0)

    #thresholding
    i_x_2[i_x_2>255] = 255
    i_x_2[i_x_2<0] = 0
    i_y_2[i_y_2 > 255] = 255
    i_y_2[i_y_2 < 0] = 0
    i_x_i_y[i_x_i_y > 255] = 255
    i_x_i_y[i_x_i_y < 0] = 0

    cv2.imshow("i_x_2", i_x_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("i_y_2", i_y_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("i_x_i_y", i_x_i_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    """ #Corner detection using Harris Corner detection method implemented in OpenCV
    dst = cv2.cornerHarris(gray_img, 2, 3, 0.04)
    print(dst.shape)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]

    #show corners detected by Harris corner detection method implemented in OpenCV
    cv2.imshow("Corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""