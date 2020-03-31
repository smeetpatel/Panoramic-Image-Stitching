import cv2
import numpy as np
import utility as utl

def project(x1,y1,H):
    #calculate projected point based on the given homography
    x2 = H[0][0]*x1 + H[0][1]*y1 + H[0][2]
    y2 = H[1][0]*x1 + H[1][1]*y1 + H[1][2]
    w = H[2][0]*x1 + H[2][1]*y1 + H[2][2]
    x2 /= w
    y2 /= w

    # check the projections with cv2.perspectiveTransform function
    # arr = np.array([[x1, y1]], dtype=np.float32)
    # dst = cv2.perspectiveTransform(arr[np.newaxis], H)

    return x2, y2

def computeInlierCount(H, matches, inlierThreshold, image_1_keypoints, image_2_keypoints):
    # get points to project and true projection
    source_points = np.float32([image_1_keypoints[match.queryIdx].pt for match in matches])
    destination_points = np.float32([image_2_keypoints[match.trainIdx].pt for match in matches])

    # project every point, calculate distance, and save all the inlier matches
    number_of_inliers = 0
    inlier_matches = []
    for point_index in range(len(source_points)):
        x, y = project(source_points[point_index][0], source_points[point_index][1], H)
        distance = utl.get_distance(destination_points[point_index][0], destination_points[point_index][1], x, y)
        if distance <= inlierThreshold:
            number_of_inliers += 1
            inlier_matches.append(matches[point_index])

    return number_of_inliers, inlier_matches

def RANSAC (matches , numMatches, numIterations, inlierThreshold, hom, homInv, image1Display, image2Display):
    # get image keypoints
    image_1_keypoints = [corner.keypoint for corner in image1Display.corners]
    image_2_keypoints = [corner.keypoint for corner in image2Display.corners]

    # declare variables to maintain maximum number of inliers for some homography and corresponding inlier matches
    maximum_inliers = 0
    best_homography_inliers = None

    # perform required iterations
    for iteration in range(numIterations):
        matches_sample = np.random.choice(matches, 4)

        source_points = np.float32([image_1_keypoints[match.queryIdx].pt for match in matches_sample])
        destination_points = np.float32([image_2_keypoints[match.trainIdx].pt for match in matches_sample])
        homography, mask = cv2.findHomography(source_points, destination_points, 0)
        if homography is None:
            continue
        number_of_inliers, inlier_matches = computeInlierCount(homography, matches, inlierThreshold, image_1_keypoints, image_2_keypoints)

        if number_of_inliers > maximum_inliers:
            maximum_inliers = number_of_inliers
            best_homography_inliers = inlier_matches

    # Calculate homography for all the inliers computed for the best homography
    source_points = np.float32([image_1_keypoints[match.queryIdx].pt for match in best_homography_inliers])
    destination_points = np.float32([image_2_keypoints[match.trainIdx].pt for match in best_homography_inliers])
    homography, mask = cv2.findHomography(source_points, destination_points, 0)
    inverse_homography = np.linalg.inv(homography)
    # inverse_homography, _ = cv2.findHomography(destination_points, source_points, 0)

    # Generate image with inlier matches
    match_image = cv2.drawMatches(image1Display.image, image_1_keypoints, image2Display.image, image_2_keypoints,
                                  matches1to2=best_homography_inliers[:10], outImg=np.array([]),
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return homography, inverse_homography, match_image