import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Class names from your dataset
CLASS_NAMES = [
    "BABY_PRODUCTS",
    "BEAUTY_HEALTH",
    "CLOTHING_ACCESSORIES_JEWELLERY",
    "ELECTRONICS",
    "GROCERY",
    "HOBBY_ARTS_STATIONERY",
    "HOME_KITCHEN_TOOLS",
    "PET_SUPPLIES",
    "SPORTS_OUTDOOR"
]

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
    model.load_state_dict(torch.load("ecommerce_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

st.title("E-Commerce Product Classifier")
st.write("Upload a product image and the model will predict its category.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item() * 100

    st.markdown("---")
    st.subheader("Prediction")
    st.success(f"**{CLASS_NAMES[predicted_idx].replace('_', ' ')}**")
    st.write(f"Confidence: {confidence:.1f}%")

    st.subheader("All class probabilities")
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
        st.progress(float(prob), text=f"{name.replace('_', ' ')}: {prob*100:.1f}%")
