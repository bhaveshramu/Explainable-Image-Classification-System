import torch
from torchvision import transforms
from PIL import Image
from model import get_model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
num_classes = 2
model = get_model(num_classes)
model.load_state_dict(torch.load("../models/model.pth", map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load image
image_path = "../dataset/test_set/cats/cat.4003.jpg"  # change this later
# pick any image file name manually and put full path here

img = Image.open(image_path)
img = transform(img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)

classes = ["cats", "dogs"]

print("Prediction:", classes[predicted.item()])