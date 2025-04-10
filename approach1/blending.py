import numpy as np
import cv2

def feather_blend(warped_img, right_img, x_offset, y_offset):
    h, w = right_img.shape[:2]
    blend = warped_img.copy()

    for y in range(h):
        for x in range(w):
            if np.any(right_img[y, x] > 0):
                alpha = 0.5  # constant blending factor
                blend[y + y_offset, x + x_offset] = (
                    alpha * right_img[y, x] + (1 - alpha) * warped_img[y + y_offset, x + x_offset]
                ).astype(np.uint8)
    return blend

def poisson_blend(warped_img, right_img, x_offset, y_offset):
    center = (right_img.shape[1] // 2 + x_offset, right_img.shape[0] // 2 + y_offset)

    mask = np.zeros(right_img.shape[:2], dtype=np.uint8)
    mask[right_img[:, :, 0] > 0] = 255

    blended = cv2.seamlessClone(
        right_img,
        warped_img,
        mask,
        center,
        cv2.NORMAL_CLONE
    )
    return blended

def optimal_seam_blend(warped_img, right_img, x_offset, y_offset):
    h, w = right_img.shape[:2]
    mask1 = (warped_img[y_offset:y_offset+h, x_offset:x_offset+w].sum(axis=2) > 0).astype(np.uint8)
    mask2 = (right_img.sum(axis=2) > 0).astype(np.uint8)
    
    overlap = mask1 & mask2

    # Distance transform for feathering weights
    dist1 = cv2.distanceTransform((mask1 * 255).astype(np.uint8), cv2.DIST_L2, 5)
    dist2 = cv2.distanceTransform((mask2 * 255).astype(np.uint8), cv2.DIST_L2, 5)

    total = dist1 + dist2 + 1e-6  # avoid division by zero
    alpha = dist1 / total

    blend_region = warped_img[y_offset:y_offset+h, x_offset:x_offset+w] * alpha[:, :, None] + right_img * (1 - alpha[:, :, None])
    result = warped_img.copy()
    result[y_offset:y_offset+h, x_offset:x_offset+w] = blend_region.astype(np.uint8)
    return result
