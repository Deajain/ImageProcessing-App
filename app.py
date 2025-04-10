import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Set Streamlit theme to light mode
st.set_page_config(page_title="Image Processing App", layout="centered")
st.markdown("""
    <style>
        body { background-color: white; color: black; }
        .stApp { background-color: white; }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Image Processing App")
st.sidebar.header("Upload Image & Select Operation")

# Upload Image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Select Image Processing Operation
    operation = st.sidebar.selectbox(
        "Choose an Image Processing Operation",
        [
            "Grayscale", "Brightness Adjustment", "Contrast Adjustment",
            "Sharpening", "Gaussian Blur", "Color Space Conversion",
            "Median Filtering", "Histogram Equalization",
            "Rotation", "Translation", "Shearing"
        ]
    )

    # Process the Image
    processed_img = img.copy()

    if operation == "Grayscale":
        processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif operation == "Brightness Adjustment":
        beta = st.sidebar.slider("Brightness", -100, 100, 50)
        processed_img = cv2.convertScaleAbs(img, beta=beta)
    elif operation == "Contrast Adjustment":
        alpha = st.sidebar.slider("Contrast", 0.5, 3.0, 1.5)
        processed_img = cv2.convertScaleAbs(img, alpha=alpha)
    elif operation == "Sharpening":
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed_img = cv2.filter2D(img, -1, sharpen_kernel)
    elif operation == "Gaussian Blur":
        processed_img = cv2.GaussianBlur(img, (5, 5), 0)
    elif operation == "Color Space Conversion":
        color_space = st.sidebar.selectbox("Choose Color Space", ["HSV", "LAB", "GRAY"])
        if color_space == "HSV":
            processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == "LAB":
            processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        elif color_space == "GRAY":
            processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif operation == "Median Filtering":
        processed_img = cv2.medianBlur(img, 5)
    elif operation == "Histogram Equalization":
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        processed_img = cv2.equalizeHist(gray_img)
    elif operation == "Rotation":
        angle = st.sidebar.slider("Rotation Angle", -180, 180, 45)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        processed_img = cv2.warpAffine(img, M, (w, h))
    elif operation == "Translation":
        tx = st.sidebar.slider("Translate X", -100, 100, 20)
        ty = st.sidebar.slider("Translate Y", -100, 100, 20)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        processed_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    elif operation == "Shearing":
        shear_factor = st.sidebar.slider("Shear Factor", -0.5, 0.5, 0.2)
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        processed_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # Show images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
    with col2:
        if len(processed_img.shape) == 2:
            st.image(processed_img, caption=f"{operation} Result", use_container_width=True, channels="GRAY")
        else:
            st.image(processed_img, caption=f"{operation} Result", use_container_width=True)

    # Download button
    if len(processed_img.shape) == 2:
        img_pil = Image.fromarray(processed_img, 'L')
    else:
        img_pil = Image.fromarray(processed_img)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    st.download_button(label="Download Processed Image", data=buf.getvalue(), file_name="processed_image.png", mime="image/png")

st.sidebar.text("Developed by Dea Jain")
