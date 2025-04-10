import os
import cv2
import torch
import kornia as K
import numpy as np
import matplotlib.pyplot as plt
import kornia.feature as KF
from kornia.contrib import ImageStitcher


def load_images(fnames):
    return [K.io.load_image(fn, K.io.ImageLoadType.RGB32)[None, ...] for fn in fnames]


left_img_path = "./images/img_1a.jpg"
right_img_path = "./images/img_1b.jpg"
os.makedirs("./output", exist_ok=True)
output_path = os.path.join("./output", "img_1a_img_1b.jpg")

imgs = load_images([left_img_path, right_img_path])

IS = ImageStitcher(KF.LoFTR(pretrained="outdoor"), estimator="ransac")

with torch.no_grad():
    out = IS(*imgs)

image_np = K.tensor_to_image(out)
image_bgr = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

cv2.imwrite(output_path, image_bgr)
