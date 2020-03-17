import cv2
import numpy as np
import utility as utl
from image import Image

if __name__ == '__main__':
    # argumentImage = 'project_images/Boxes.png'
    # argumentImage = 'project_images/Rainier1.png'
    argumentImage = 'project_images/Rainier2.png'

    boxes = Image(argumentImage)
    boxes.detect_harris_corners()