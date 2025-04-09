
import sys
print(sys.executable)
import os
from PIL import Image
import torch
import torchvision.transforms as T
from stitching import stitch_background, panorama

# Utility function to load and convert image to torch tensor (uint8)
def load_image_as_tensor(image_path):
    transform = T.ToTensor()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image) * 255
    return tensor.to(torch.uint8)

def main():
    # ====== Task 1: stitch_background ======
    print("Running Task 1: stitch_background...")

    img1_path = "images/stitching/t1_1.png"
    img2_path = "images/stitching/t1_2.png"

    img1 = load_image_as_tensor(img1_path)
    img2 = load_image_as_tensor(img2_path)

    imgs_task1 = {
        "t1_1.png": img1,
        "t1_2.png": img2
    }

    result_task1 = stitch_background(imgs_task1)

    # Save result
    result_img1 = T.ToPILImage()(result_task1)
    result_img1.save("output/stitched_background.png")
    print("Saved stitched background to output/stitched_background.png")

    # ====== Task 2: panorama ======
    print("Running Task 2: panorama...")

    # List of panorama images
    panorama_paths = [
        "images/panaroma/t2_5.jpeg",
        "images/panaroma/t2_6.jpeg",
        "images/panaroma/t2_7.jpeg"
    ]

    imgs_task2 = {
        os.path.basename(path): load_image_as_tensor(path)
        for path in panorama_paths
    }

    result_task2, overlap = panorama(imgs_task2)

    # Save result
    result_img2 = T.ToPILImage()(result_task2)
    result_img2.save("output/panorama_result.png")
    print("Saved panorama result to output/panorama_result.png")

    print("Overlap matrix:\n", overlap)

if __name__ == "__main__":
    main()
