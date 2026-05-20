import numpy as np
import cv2  # OpenCV for image loading and processing
import joblib
import tensorflow as tf
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import os
import base64
from io import BytesIO

# Set the device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MobileNetModel(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)
    
fracture_model = MobileNetModel(num_classes=2)
fracture_model.load_state_dict(torch.load("manageApp/models/mobilenet.pt", map_location=device))
fracture_model = fracture_model.to(device)
fracture_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

feature_extractor = tf.keras.models.load_model('manageApp/models/mobilenet_feature_extractor.h5')
rf_model = joblib.load('manageApp/models/random_forest_classifier.pkl')

class_names = [
    'Avulsion fracture',
    'Comminuted fracture',
    'Fracture Dislocation',
    'Greenstick fracture',
    'Hairline Fracture',
    'Impacted fracture',
    'Longitudinal fracture',
    'Oblique fracture',
    'Pathological fracture',
    'Spiral Fracture'
]

def map_prediction_to_label(prediction):
    label_mapping = {0: "No Fracture", 1: "Fracture"}
    return label_mapping.get(prediction, "Unknown")

def predict_fracture(image):
    image = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = fracture_model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()


def preprocess_image_for_keras(image):
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def classify_fracture_type(image):
    new_image = preprocess_image_for_keras(image)
    features = feature_extractor.predict(new_image)
    features = features.reshape(features.shape[0], -1)
    prediction_index = rf_model.predict(features)[0]
    return class_names[prediction_index]


def decode_base64_to_image(base64_string):
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        missing = len(base64_string) % 4
        if missing != 0:
            base64_string += "=" * (4 - missing)


        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
        
        if img is None:
            raise ValueError("Decoded image is empty or invalid.")
        
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def main(base64_image):
    image = decode_base64_to_image(base64_image)
    
    if image is None:
        return "Failed to decode the image."
    
    try:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
        fracture_prediction = predict_fracture(pil_image)
        predicted_label = map_prediction_to_label(fracture_prediction)

        if predicted_label == "Fracture":
            fracture_type = classify_fracture_type(image)
            return f"Detected Fracture Type: {fracture_type}"
        else:
            return "No Fracture Detected."
    except Exception as e:
        print(f"Error during prediction: {e}")


