import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Streamlit UI
st.title("Image Processing App")
st.sidebar.header("Upload Image & Select Operation")

# Upload Image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)
    
    # Display original image
    st.image(img, caption="Original Image", use_column_width=True)

    # Select Image Processing Operation
    operation = st.sidebar.selectbox(
        "Choose an Image Processing Operation",
        [
            "Grayscale", "Brightness Adjustment", "Contrast Adjustment",
            "Histogram Equalization", "Gamma Correction", "Sharpening",
            "Gaussian Blur", "Median Filtering", "Bilateral Filtering",
            "Resizing", "Rotation", "Translation", "Shearing",
            "Color Space Conversion", "Color Filtering"
        ]
    )

    # Process the Image
    if operation == "Grayscale":
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        st.image(gray_img, caption="Grayscale Image", use_column_width=True, channels="GRAY")

    elif operation == "Brightness Adjustment":
        beta = st.sidebar.slider("Brightness", -100, 100, 50)
        bright_img = cv2.convertScaleAbs(img, beta=beta)
        st.image(bright_img, caption="Brightness Adjusted", use_column_width=True)

    elif operation == "Contrast Adjustment":
        alpha = st.sidebar.slider("Contrast", 0.5, 3.0, 1.5)
        contrast_img = cv2.convertScaleAbs(img, alpha=alpha)
        st.image(contrast_img, caption="Contrast Adjusted", use_column_width=True)

    elif operation == "Histogram Equalization":
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        equalized_img = cv2.equalizeHist(gray_img)
        st.image(equalized_img, caption="Histogram Equalized", use_column_width=True, channels="GRAY")

    elif operation == "Gamma Correction":
        gamma = st.sidebar.slider("Gamma", 0.1, 3.0, 1.0)
        gamma_corrected = np.power(img / 255.0, gamma) * 255.0
        gamma_corrected = np.uint8(gamma_corrected)
        st.image(gamma_corrected, caption="Gamma Corrected", use_column_width=True)

    elif operation == "Sharpening":
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(img, -1, sharpen_kernel)
        st.image(sharpened, caption="Sharpened Image", use_column_width=True)

    elif operation == "Gaussian Blur":
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        st.image(blurred, caption="Gaussian Blurred", use_column_width=True)

    elif operation == "Median Filtering":
        median_filtered = cv2.medianBlur(img, 5)
        st.image(median_filtered, caption="Median Filtered", use_column_width=True)

    elif operation == "Bilateral Filtering":
        bilateral_filtered = cv2.bilateralFilter(img, 9, 75, 75)
        st.image(bilateral_filtered, caption="Bilateral Filtered", use_column_width=True)

    elif operation == "Resizing":
        scale_percent = st.sidebar.slider("Resize Scale (%)", 10, 200, 50)
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        resized = cv2.resize(img, (width, height))
        st.image(resized, caption="Resized Image", use_column_width=True)

    elif operation == "Rotation":
        angle = st.sidebar.slider("Rotation Angle", -180, 180, 45)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        st.image(rotated, caption="Rotated Image", use_column_width=True)

    elif operation == "Translation":
        tx = st.sidebar.slider("Translate X", -100, 100, 20)
        ty = st.sidebar.slider("Translate Y", -100, 100, 20)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        st.image(translated, caption="Translated Image", use_column_width=True)

    elif operation == "Shearing":
        shear_factor = st.sidebar.slider("Shear Factor", -0.5, 0.5, 0.2)
        M = np.float32([[1, shear_factor, 0], [shear_factor, 1, 0]])
        sheared = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        st.image(sheared, caption="Sheared Image", use_column_width=True)

    elif operation == "Color Space Conversion":
        color_space = st.sidebar.selectbox("Choose Color Space", ["HSV", "LAB", "GRAY"])
        if color_space == "HSV":
            converted = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == "LAB":
            converted = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        elif color_space == "GRAY":
            converted = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        st.image(converted, caption=f"{color_space} Image", use_column_width=True)

    elif operation == "Color Filtering":
        st.sidebar.subheader("Color Filtering Options")
        lower_h = st.sidebar.slider("Lower Hue", 0, 180, 0)
        upper_h = st.sidebar.slider("Upper Hue", 0, 180, 10)
        lower_s = st.sidebar.slider("Lower Saturation", 0, 255, 100)
        upper_s = st.sidebar.slider("Upper Saturation", 0, 255, 255)
        lower_v = st.sidebar.slider("Lower Value", 0, 255, 100)
        upper_v = st.sidebar.slider("Upper Value", 0, 255, 255)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_bound = np.array([lower_h, lower_s, lower_v])
        upper_bound = np.array([upper_h, upper_s, upper_v])
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        filtered_img = cv2.bitwise_and(img, img, mask=mask)

        st.image(img, caption="Original Image", use_container_width=True)

st.sidebar.text("Developed with ❤️ by Dea Jain")
