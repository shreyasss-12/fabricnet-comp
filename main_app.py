import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the best model (Fold 3)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('fabricnet_best_fold_3.keras')
    return model

model = load_model()

# Preprocessing function
# Adjusted to match the model's expected input size: (120, 120, 3)
def preprocess_image(image):
    image = image.resize((120, 120))  # Correct size as per model requirement
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
# Updated to handle multi-output model
def predict_composition(image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predictions = [float(p[0][0]) for p in predictions]  # extract each scalar
    return predictions

# Streamlit UI
st.title("🧵 Fabric Composition Predictor")

st.write("Upload an image of a fabric to predict its composition.")

uploaded_file = st.file_uploader("Upload your Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    predictions = predict_composition(image)

    # Updated labels for 6 classes
    labels = ['Cotton', 'Wool', 'Polyester', 'Linen', 'Silk', 'Other Fibres']

    st.write("### Predicted Fabric Composition:")

    if len(predictions) == len(labels):
        for i, label in enumerate(labels):
            st.write(f"**{label}:** {predictions[i] * 100:.2f}%")
        st.success("Prediction completed successfully!")
    else:
        st.error(f"Model output size ({len(predictions)}) does not match expected number of classes ({len(labels)}). Please check the model and label list.")

st.markdown("---")
st.markdown("Developed by Shreyas S.")