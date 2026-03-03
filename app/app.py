import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import sys
import os

# Add src folder to path
sys.path.append(os.path.abspath("../src"))

from model import get_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Page config
st.set_page_config(page_title="Explainable Image Classification System", layout="wide")

st.title("Explainable Image Classification System")
st.markdown("""
This system uses a pretrained **ResNet18 model** to classify images 
and applies **Grad-CAM** to visualize the regions influencing the prediction.
""")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
num_classes = 2
model = get_model(num_classes)
model.load_state_dict(torch.load("../models/model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["Cats", "Dogs"]

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width=400)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item() * 100

    # Grad-CAM
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted.item())]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]

    rgb_array = np.array(image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_array, grayscale_cam, use_rgb=True)

    with col2:
        st.subheader("Grad-CAM Heatmap")
        st.image(visualization, width=400)

    st.markdown("---")
    st.subheader("Prediction Results")

    st.write(f"**Predicted Class:** {predicted_class}")
    st.progress(int(confidence_score))
    st.write(f"Confidence: {confidence_score:.2f}%")

    st.markdown("---")
    st.subheader("Explanation Summary")

    if predicted_class == "Cats":
        st.write(
            "The model classified this image as a Cat based on detected facial structure, "
            "fur texture, and ear positioning typical of feline features. "
            "Highlighted regions represent areas that strongly influenced the decision."
        )
    else:
        st.write(
            "The model classified this image as a Dog due to detection of snout shape, "
            "fur distribution, and head structure commonly associated with canine features. "
            "Highlighted regions indicate the most influential areas in the prediction."
        )


    st.markdown("---")
    st.subheader("Model Details")
    st.write("Architecture: ResNet18 (Pretrained on ImageNet)")
    st.write("Framework: PyTorch")
    st.write("Explainability Method: Grad-CAM")
    st.write("Deployment: Streamlit Web Application")

st.markdown("---")
st.caption("Final Year Engineering Project | Explainable AI System")