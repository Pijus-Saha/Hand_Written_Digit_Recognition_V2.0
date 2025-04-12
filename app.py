import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("digit_recognition_model.h5")

# Define the image preprocessing function
# This function converts the image to grayscale, resizes it to 28x28 pixels,
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to MNIST format
    image = np.array(image)

    # Apply adaptive threshold to binarize
    threshold = 200
    image = np.where(image > threshold, 255, 0).astype(np.uint8)

    # Invert if digit is black on white
    if np.mean(image) > 127:
        image = 255 - image

    image = image / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)  # CNN input
    return image


# Streamlit UI
st.title("📝 Handwritten Digit Recognition V2.0")
st.write("Upload a handwritten digit image, and the model will predict the digit.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    
    import matplotlib.pyplot as plt
    st.subheader("Processed Image")
    st.image(processed_image.reshape(28, 28), width=150, clamp=True)


    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    # Show prediction result
    st.success(f"**Predicted Digit:** {predicted_digit}")
    st.write("Confidence Scores:")
    for i, score in enumerate(prediction[0]):
        st.write(f"Digit {i}: {score*100:.2f}%")
