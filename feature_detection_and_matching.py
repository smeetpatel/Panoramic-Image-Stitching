import cv2
import utility as utl
from image import Image


"""
This module demonstrates the following:

1) Working of 'Harris corner detector' implemented by me. 
    - Input files: 
        i) 'image_sets/project_images/Boxes.png'  
        ii) 'image_sets/project_images/Rainier1.png' 
        iii) 'image_sets/project_images/Rainier2.png'
    - Corresponding output files with detected Harris Corners are: 
        i) 'Results/project_images/1a.png'  
        ii) 'Results/project_images/1b.png'
        iii) 'Results/project_images/1c.png'
        
2) Calculates the rotation and illumination invariant SIFT descriptors for the detected keypoints.

3) Finds appropriate matches between two images based on the keypoints detected.
    - Input files: 
        i) 'image_sets/project_images/Rainier1.png' 
        ii) 'image_sets/project_images/Rainier2.png'
    - Output files:
        i) 'Results/project_images/2.png' contains the image with matches between the input images.
"""
if __name__ == '__main__':
    corner_detection_images = ['project_images/Boxes.png', 'project_images/Rainier1.png', 'project_images/Rainier2.png']
    images = []
    for argumentImage in corner_detection_images:
        image = Image(argumentImage)
        image.detect_harris_corners()
        images.append(image)

    matching_image = utl.get_matches(images[1], images[2])
    cv2.imwrite("Results/project_images/2.png", matching_image)