import cv2
import numpy as np
import random
import time
import os
import sys

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    key_points1, descriptor1, key_points2, descriptor2 = get_keypoint(left_img, right_img)
    good_matches = match_keypoint(key_points1, key_points2, descriptor1, descriptor2)
    final_H = ransac(good_matches)

    rows1, cols1 = right_img.shape[:2]
    rows2, cols2 = left_img.shape[:2]

    points1 =  np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    points  =  np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    points2 =  cv2.perspectiveTransform(points, final_H)
    list_of_points = np.concatenate((points1,points2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    H_translation = (np.array([[1, 0, (-x_min)], [0, 1, (-y_min)], [0, 0, 1]])).dot(final_H)

    output_img = cv2.warpPerspective(left_img, H_translation, (x_max-x_min, y_max-y_min))
    output_img[(-y_min):rows1+(-y_min), (-x_min):cols1+(-x_min)] = right_img
    result_img = output_img
    return result_img
    
def get_keypoint(left_img, right_img):
    l_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    r_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    print("Using Harris Corner Detection for keypoint detection.")

    def harris_corners(gray):
        corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        corners = cv2.dilate(corners, None)  # Optional: to enhance corner points
        threshold = 0.01 * corners.max()
        keypoints = np.argwhere(corners > threshold)
        keypoints = [cv2.KeyPoint(float(pt[1]), float(pt[0]), 1) for pt in keypoints]
        return keypoints

    kp1 = harris_corners(l_img)
    kp2 = harris_corners(r_img)

    print(f"Harris corners detected: Left = {len(kp1)}, Right = {len(kp2)}")

    # Extract 9x9 patch descriptors
    def extract_patches(img, keypoints, patch_size=9):
        half = patch_size // 2
        descriptors = []
        valid_keypoints = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if y - half < 0 or y + half >= img.shape[0] or x - half < 0 or x + half >= img.shape[1]:
                continue
            patch = img[y - half:y + half + 1, x - half:x + half + 1]
            descriptors.append(patch.flatten())
            valid_keypoints.append(kp)
        return valid_keypoints, np.array(descriptors)

    kp1, desc1 = extract_patches(l_img, kp1)
    kp2, desc2 = extract_patches(r_img, kp2)

    return kp1, desc1, kp2, desc2

def match_keypoint(key_points1, key_points2, descriptor1, descriptor2):
    print("Matching descriptors using SSD + ratio test.")
    matches = []
    for i, d1 in enumerate(descriptor1):
        distances = np.linalg.norm(descriptor2 - d1, axis=1)
        idx = np.argsort(distances)
        if distances[idx[0]] < 0.75 * distances[idx[1]]:  # ratio test
            pt1 = key_points1[i].pt
            pt2 = key_points2[idx[0]].pt
            matches.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    print(f"Good matches found: {len(matches)}")
    return matches

def homography(points):
    A = []
    for pt in points:
      x, y = pt[0], pt[1]
      X, Y = pt[2], pt[3]
      A.append([x, y, 1, 0, 0, 0, -1 * X * x, -1 * X * y, -1 * X])
      A.append([0, 0, 0, x, y, 1, -1 * Y * x, -1 * Y * y, -1 * Y])

    A = np.array(A)
    u, s, vh = np.linalg.svd(A)
    H = (vh[-1, :].reshape(3, 3))
    H = H/ H[2, 2]
    return H

def ransac(good_pts):
    best_inliers = []
    final_H = []
    t=5
    for i in range(5000):
        random_pts = random.choices(good_pts, k=4)
        H = homography(random_pts)
        inliers = []
        for pt in good_pts:
            p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
            p_1 = np.array([pt[2], pt[3], 1]).reshape(3, 1)
            Hp = np.dot(H, p)
            Hp = Hp / Hp[2]
            dist = np.linalg.norm(p_1 - Hp)

            if dist < t: inliers.append(pt)

        if len(inliers) > len(best_inliers):
            best_inliers,final_H = inliers,H
    return final_H

if __name__ == "__main__":
    start_time = time.time()
    
    left_img_name = sys.argv[1].split('.')[0]
    right_img_name = sys.argv[2].split('.')[0]

    left_img_path = os.path.join('./images', sys.argv[1])
    right_img_path = os.path.join('./images', sys.argv[2])

    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)

    print("Starting stitching process...")
    result_img = solution(left_img, right_img)
    
    print("Stitching completed.")
    print("Result image shape:", result_img.shape)
    output_path = f'./output/harris/{left_img_name}_{right_img_name}.jpg'

    cv2.imwrite(output_path, result_img)
    
    end_time = time.time()
    print("Stitching took {:.2f} seconds.".format(end_time - start_time))
    print("Stitching completed.")
