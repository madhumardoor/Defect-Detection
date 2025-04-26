import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, 2)  # 2 classes: Normal, Defect

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Instantiate and load the model
model = CNN()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Streamlit UI
st.title("🛠️ Defect Detection System")
st.write("Upload an image to check if it's **normal** or **defective**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    img = transform(image)
    img = img.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        label = predicted.item()

    # Display the result
    classes = ['Normal', 'Defect']
    st.subheader(f"Prediction: **{classes[label]}**")

    if label == 0:
        st.success("✅ No Defect Detected")
    else:
        st.error("⚠️ Defect Detected")
