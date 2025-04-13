import cv2
import numpy as np
import random
import os
import sys
import time

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

    orb = cv2.ORB_create(nfeatures=2000)

    print("Using ORB for keypoint detection and description.")

    key_points1, descriptor1 = orb.detectAndCompute(l_img, None)
    key_points2, descriptor2 = orb.detectAndCompute(r_img, None)

    print("Number of keypoints in left image:", len(key_points1))
    print("Number of keypoints in right image:", len(key_points2))

    return key_points1, descriptor1, key_points2, descriptor2

def match_keypoint(key_points1, key_points2, descriptor1, descriptor2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            left_pt = key_points1[m.queryIdx].pt
            right_pt = key_points2[m.trainIdx].pt
            good_matches.append([left_pt[0], left_pt[1], right_pt[0], right_pt[1]])
    return good_matches

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

    left_img_path = os.path.join('../images', sys.argv[1])
    right_img_path = os.path.join('../images', sys.argv[2])

    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)

    print("Starting stitching process...")
    result_img = solution(left_img, right_img)
    
    print("Stitching completed.")
    print("Result image shape:", result_img.shape)
    output_path = f'./output/orb/{left_img_name}_{right_img_name}.jpg'

    cv2.imwrite(output_path, result_img)
    
    end_time = time.time()
    print("Stitching took {:.2f} seconds.".format(end_time - start_time))
    print("Stitching completed.")
