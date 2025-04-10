from pathlib import Path
import numpy as np
import matplotlib.cm as cm
import torch
import cv2 as cv
import os

from models.matching import Matching
from models.utils import read_image, make_matching_plot, AverageTimer

torch.set_grad_enabled(False)

left_img_path = '../images/img_1a.jpg'
right_img_path = '../images/img_1b.jpg'
output_path = './output/matches/v0/img_1a_img_1b.jpg'
stitched_output_path = './output/v0/img_1a_img_1b.jpg'

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

timer = AverageTimer(newline=True)
image0, inp0, scales0 = read_image(Path(left_img_path), device, resize, 0, resize_float)
image1, inp1, scales1 = read_image(Path(right_img_path), device, resize, 0, resize_float)

if image0 is None or image1 is None:
    raise ValueError('Failed to load one or both images.')

pred = matching({'image0': inp0, 'image1': inp1})
pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
matches, conf = pred['matches0'], pred['matching_scores0']
timer.update('matching')

valid = matches > -1
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
mconf = conf[valid]

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

make_matching_plot(
    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
    text, output_path, show_keypoints,
    fast_viz, opencv_display, 'Matches', small_text
)

print(f'Match visualization saved to: {output_path}')

os.makedirs(os.path.dirname(stitched_output_path), exist_ok=True)

im_left = cv.imread(left_img_path, cv.IMREAD_COLOR)
im_right = cv.imread(right_img_path, cv.IMREAD_COLOR)

H, status = cv.findHomography(mkpts0, mkpts1, cv.RANSAC, 5.0)

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
    translation[1]:translation[1]+h1,
    translation[0]:translation[0]+w1
] = im_left

cv.imwrite(stitched_output_path, warped_right)
print(f"Stitched image saved to: {stitched_output_path}")
