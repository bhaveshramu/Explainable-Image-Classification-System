import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from model import get_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
num_classes = 2
model = get_model(num_classes)
model.load_state_dict(torch.load("../models/model.pth", map_location=device))
model.to(device)
model.eval()

# Target layer (last convolution layer of ResNet18)
target_layers = [model.layer4[-1]]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Select test image (CHANGE THIS PATH)
image_path = "../dataset/test_set/dogs/dog.4003.jpg"

# Load image
rgb_img = Image.open(image_path).convert("RGB")
rgb_img = rgb_img.resize((224, 224))
rgb_array = np.array(rgb_img) / 255.0

input_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

# Predict class
with torch.no_grad():
    outputs = model(input_tensor)
    predicted_class = torch.argmax(outputs, dim=1).item()

# Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers)

targets = [ClassifierOutputTarget(predicted_class)]
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0]

visualization = show_cam_on_image(rgb_array, grayscale_cam, use_rgb=True)

# Save result
cv2.imwrite("../outputs/heatmap.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

classes=["Cat","Dog"]
print(f"Prediction class index:{predicted_class} and class name:{classes[predicted_class]}")
#print("Prediction class name:", classes[predicted_class])
print("Heatmap saved to outputs/heatmap.jpg")