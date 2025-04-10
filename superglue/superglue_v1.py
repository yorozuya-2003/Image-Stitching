from pathlib import Path
import numpy as np
import matplotlib.cm as cm
import torch
import cv2 as cv
import os
import tempfile

from models.matching import Matching
from models.utils import read_image, make_matching_plot


def stitch_images(left_image_file, right_image_file, save_matches=False, matches_output_path=None):
    """
    Stitches two images using SuperGlue matching.
    """
    resize = [-1]
    resize_float = True
    superglue_weights = 'outdoor'
    max_keypoints = 2048
    keypoint_threshold = 0.05
    nms_radius = 5
    sinkhorn_iterations = 20
    match_threshold = 0.9
    show_keypoints = False
    fast_viz = False
    opencv_display = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on device: {device}')

    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints,
        },
        'superglue': {
            'weights': superglue_weights,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }

    matching = Matching(config).eval().to(device)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_left, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_right:

        tmp_left.write(left_image_file.read())
        tmp_right.write(right_image_file.read())

        left_path = Path(tmp_left.name)
        right_path = Path(tmp_right.name)

    image0, inp0, scales0 = read_image(left_path, device, resize, 0, resize_float)
    image1, inp1, scales1 = read_image(right_path, device, resize, 0, resize_float)

    if image0 is None or image1 is None:
        raise ValueError('Failed to load one or both images.')

    with torch.no_grad():
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    if save_matches and matches_output_path:
        color = cm.jet(mconf)
        text = [
            'SuperGlue Matching',
            f'Keypoints: {len(kpts0)}:{len(kpts1)}',
            f'Matches: {len(mkpts0)}'
        ]
        small_text = [
            f'Keypoint Threshold: {keypoint_threshold:.4f}',
            f'Match Threshold: {match_threshold:.2f}',
        ]

        os.makedirs(os.path.dirname(matches_output_path), exist_ok=True)
        make_matching_plot(
            image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
            text, matches_output_path, show_keypoints,
            fast_viz, opencv_display, 'Matches', small_text
        )
        print(f'Match visualization saved to: {matches_output_path}')

    # Homography and stitching
    matched_points = np.hstack((mkpts0, mkpts1))
    H = _ransac(matched_points)

    im_left = cv.imread(str(left_path), cv.IMREAD_COLOR)
    im_right = cv.imread(str(right_path), cv.IMREAD_COLOR)

    h1, w1 = im_left.shape[:2]
    h2, w2 = im_right.shape[:2]

    corners_right = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv.perspectiveTransform(corners_right, np.linalg.inv(H))

    corners_left = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    all_corners = np.concatenate((warped_corners, corners_left), axis=0)

    [x_min, y_min] = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    [x_max, y_max] = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    translation = [-x_min, -y_min]
    translate_mat = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

    output_size = (x_max - x_min, y_max - y_min)
    warped_right = cv.warpPerspective(im_right, translate_mat @ np.linalg.inv(H), output_size)

    warped_right[
        translation[1]:translation[1] + h1,
        translation[0]:translation[0] + w1
    ] = im_left

    return warped_right


def _homography(points):
    A = []
    for pt in points:
        x, y = pt[0], pt[1]
        X, Y = pt[2], pt[3]
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])
    A = np.array(A)
    _, _, vh = np.linalg.svd(A)
    H = vh[-1].reshape(3, 3)
    return H / H[2, 2]


def _ransac(good_pts, iterations=5000, threshold=5):
    best_inliers = []
    final_H = None
    for _ in range(iterations):
        sample = good_pts[np.random.choice(len(good_pts), 4, replace=False)]
        H = _homography(sample)
        inliers = []
        for pt in good_pts:
            p1 = np.array([pt[0], pt[1], 1.0]).reshape(3, 1)
            p2 = np.array([pt[2], pt[3], 1.0]).reshape(3, 1)
            Hp1 = H @ p1
            Hp1 /= Hp1[2]
            dist = np.linalg.norm(p2 - Hp1)
            if dist < threshold:
                inliers.append(pt)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            final_H = H
    return final_H


def main():
    left_img_path = '../images/img_1a.jpg'
    right_img_path = '../images/img_1b.jpg'
    output_path = './output/v1/img_1a_img_1b.jpg'
    matches_path = './output/matches/v1/img_1a_img_1b.jpg'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(left_img_path, 'rb') as f_left, open(right_img_path, 'rb') as f_right:
        stitched = stitch_images(
            f_left, f_right,
            save_matches=True,
            matches_output_path=matches_path
        )
        cv.imwrite(output_path, stitched)
        print(f"Stitched image saved to: {output_path}")


if __name__ == "__main__":
    main()
