import cv2
import numpy as np

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
