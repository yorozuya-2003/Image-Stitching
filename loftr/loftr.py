import os
import cv2
import torch
import kornia as K
import numpy as np
import kornia.feature as KF
from kornia.contrib import ImageStitcher
import tempfile


def _load_image_from_filelike(file):
    if isinstance(file, str):
        return K.io.load_image(file, K.io.ImageLoadType.RGB32)[None, ...]
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
        return K.io.load_image(tmp_path, K.io.ImageLoadType.RGB32)[None, ...]


def stitch_images_kornia(left_image_file, right_image_file):
    """
    Stitch two images using Kornia's LoFTR and ImageStitcher.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_left = _load_image_from_filelike(left_image_file).to(device)
    img_right = _load_image_from_filelike(right_image_file).to(device)

    stitcher = ImageStitcher(KF.LoFTR(pretrained="outdoor"), estimator="ransac").to(device)

    with torch.no_grad():
        result = stitcher(img_left, img_right)

    stitched_image_rgb = K.tensor_to_image(result)
    stitched_image_bgr = cv2.cvtColor((stitched_image_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    return stitched_image_bgr


def main():
    left_img_path = "../images/img_1a.jpg"
    right_img_path = "../images/img_1b.jpg"
    output_path = "./output/img_1a_img_1b.jpg"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(left_img_path, 'rb') as f_left, open(right_img_path, 'rb') as f_right:
        stitched = stitch_images_kornia(f_left, f_right)
        cv2.imwrite(output_path, stitched)
        print(f"Stitched image saved to: {output_path}")


if __name__ == "__main__":
    main()