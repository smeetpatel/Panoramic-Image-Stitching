# Panoramic Image Stitching
This repository holds the code for stitching multiple images of the same scene to create panoramas. 

###### Method
The process followed to stitch multiple images of the scene to create Panoramas is as followes:
- Detect features/corners for the given images
  - Implemented 'Harris corner detector' to detect useful features and implemented rotation and illumination invariant SIFT descriptor to represent the corners.
  - Used OpenCV's implementation of SIFT keypoint detector and descriptor when images have variance in scale.
- Match features across images
  - Implemented a mechanism to match features between the images using 'Sum of squared distance(SSD)'.
  - Applied 'Ratio test' to improve the matches.
- Implemented 'Random Sample Consensus(RANSAC)' to find inlier matches.
- Implemented the algorithm to 'stitch' the images based on inlier matches.
  - Found the appropriate image to stitch next, calculated the size of the new stitched image, aligned the photographs, then warped and blended the images.
  
###### Technologies
- Python
- OpenCV
- Numpy
