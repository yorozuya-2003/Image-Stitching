import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image

from traditional_cv.stitcher import PanoramaStitcher
from superglue.stitcher import stitch_images

st.set_page_config(page_title="Image Stitcher")
with open("assets/styles.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

detector_config = {
    'sift': {
        'matcher_types': ['bf', 'flann'],
        'blending_types': ['none', 'feather', 'poisson', 'optimal_seam']
    },
    'orb': {
        'matcher_types': ['bf', 'flann'],
        'blending_types': ['none', 'feather', 'optimal_seam']
    },
    'fast+brief': {
        'matcher_types': ['bf'],
        'blending_types': ['none', 'optimal_seam']
    },
    'harris': {
        'matcher_types': ['bf'],
        'blending_types': ['none', 'optimal_seam']
    }
}

detector_labels = {
    'sift': 'SIFT',
    'orb': 'ORB',
    'fast+brief': 'FAST + BRIEF',
    'harris': 'Harris'
}
matcher_labels = {
    'bf': 'Brute Force (BF)',
    'flann': 'FLANN'
}
blending_labels = {
    'none': 'None',
    'feather': 'Feather',
    'poisson': 'Poisson',
    'optimal_seam': 'Optimal Seam'
}
detector_label_to_key = {v: k for k, v in detector_labels.items()}
matcher_label_to_key = {v: k for k, v in matcher_labels.items()}
blending_label_to_key = {v: k for k, v in blending_labels.items()}

SAMPLE_DIR = "./images"

def load_image(image_file):
    if isinstance(image_file, str):
        img = Image.open(os.path.join(SAMPLE_DIR, image_file)).convert("RGB")
    else:
        img = Image.open(image_file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

st.title("🧵 Image Stitcher")

sample_image_pairs = {
    "Sample - 1": ("img_1a.jpg", "img_1b.jpg"),
    "Sample - 2": ("img_2a.jpg", "img_2b.jpg"),
    "Sample - 3": ("img_3a.jpg", "img_3b.jpg"),
    "Sample - 4": ("img_4a.jpg", "img_4b.jpg"),
    "Sample - 5": ("img_5a.jpg", "img_5b.jpg"),
    "Sample - 6": ("img_6a.jpg", "img_6b.jpg"),
    "Sample - 7": ("img_7a.png", "img_7b.png"),
    "Sample - 8": ("img_8a.jpg", "img_8b.jpg"),
    "Sample - 9": ("img_9a.jpg", "img_9b.jpg")
}

source_mode = st.radio("Choose Image Source", ["Upload Images", "Use Sample Images"], horizontal=True)

left_img, right_img = None, None
left_col, right_col = st.columns(2)

if source_mode == "Upload Images":
    with left_col:
        left_file = st.file_uploader("Upload Left Image", type=["jpg", "jpeg", "png"], key="left")
        if left_file:
            left_img = load_image(left_file)
            st.image(left_file, caption="Left Image", use_container_width=True)

    with right_col:
        right_file = st.file_uploader("Upload Right Image", type=["jpg", "jpeg", "png"], key="right")
        if right_file:
            right_img = load_image(right_file)
            st.image(right_file, caption="Right Image", use_container_width=True)

else:
    selected_pair_name = st.selectbox("Select Sample Image Pair", list(sample_image_pairs.keys()))
    selected_pair = sample_image_pairs[selected_pair_name]

    with left_col:
        left_img = load_image(selected_pair[0])
        st.image(os.path.join(SAMPLE_DIR, selected_pair[0]), caption=f"Left Image", use_container_width=True)

    with right_col:
        right_img = load_image(selected_pair[1])
        st.image(os.path.join(SAMPLE_DIR, selected_pair[1]), caption=f"Right Image", use_container_width=True)

method = st.selectbox("Select Approach", ["Traditional CV", "SuperGlue"])

if method == "Traditional CV":
    cols = st.columns(3)
    with cols[0]:
        detector_label = st.selectbox("Select Feature Detector", list(detector_labels.values()), key='detector_type')
        detector_type = detector_label_to_key[detector_label]

        if detector_type:
            with cols[1]:
                available_blendings = detector_config[detector_type]['blending_types']
                blending_label = st.selectbox(
                    "Select Blending Type",
                    [blending_labels[b] for b in available_blendings],
                    key='blending_type'
                )
                blending_type = blending_label_to_key[blending_label]

if st.button("Stitch Images"):
    if left_img is not None and right_img is not None:
        with st.spinner("🔄 Stitching in progress..."):
            if method == "Traditional CV":
                stitcher = PanoramaStitcher()
                result_img, exec_time = stitcher.stitch(
                    left_img, right_img,
                    detector=detector_type,
                    matcher="BF",
                    blending_type=blending_type,
                    show=False,
                    return_time=True
                )
                st.success(f"✅ Stitching completed in {exec_time:.2f} seconds")
            else:
                is_upload = source_mode == "Upload Images"
                if is_upload:
                    left_file.seek(0)
                    right_file.seek(0)
                    result_img = stitch_images(left_file, right_file, save_matches=False)
                else:
                    with open(os.path.join(SAMPLE_DIR, selected_pair[0]), 'rb') as lf, \
                         open(os.path.join(SAMPLE_DIR, selected_pair[1]), 'rb') as rf:
                        result_img = stitch_images(lf, rf, save_matches=False)
                st.success("✅ Stitching completed using SuperGlue")

            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="🧵 Stitched Image", use_container_width=True)
    else:
        st.warning("⚠️ Please upload or select both left and right images.")