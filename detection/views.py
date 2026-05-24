from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.conf import settings
from .forms import ImageUploadForm
from PIL import Image
import os
import json
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score as accuracy
from django.core.files.storage import default_storage
import uuid

# Define your model class again (same architecture as used during training)
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        def ConvBlock(in_c, out_c, pool=False):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                      nn.BatchNorm2d(out_c),
                      nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(4))
            return nn.Sequential(*layers)
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))

# ---------------- Step 1: Define Base Class ----------------
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Forward pass
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Forward pass
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_accuracy': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accuracies = [x['val_accuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accuracies).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_accuracy': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch {epoch}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_accuracy']:.4f}")

# ---------------- Step 2: Define ResNet9 Model ----------------
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Define class names in the same order as used during training
disease_classes = [
    'Apple Scab', 'Apple Black Rot', 'Apple Cedar Apple Rust', 'Apple Healthy',
    'Blueberry Healthy', 'Cherry Healthy', 'Cherry Powdery Mildew',
    'Corn Cercospora Leaf Spot Gray Leaf Spot', 'Corn Common Rust', 'Corn Healthy',
    'Corn Northern Leaf Blight', 'Grape Black Rot', 'Grape Esca', 'Grape Healthy',
    'Grape Leaf Blight', 'Orange Huanglongbing', 'Peach Bacterial Spot',
    'Peach Healthy', 'Pepper Bell Bacterial Spot', 'Pepper Bell Healthy', 'Potato Early Blight',
    'Potato Healthy', 'Potato Late Blight', 'Raspberry Healthy', 'Soybean Healthy', 'Squash Powdery Mildew',
    'Strawberry Healthy', 'Strawberry Leaf Scorch', 'Tomato Bacterial Spot', 'Tomato Early Blight',
    'Tomato Healthy', 'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites', 'Tomato Target Spot', 'Tomato Mosaic Virus', 'Tomato Yellow Leaf Curl Virus'
]

# Fertilizer recommendation based on disease
fertilizer_recommendations = {
    "Scab": "Use a fungicide like Captan or Mancozeb. Improve air circulation.",
    "Black Rot": "Apply copper-based fungicides. Remove infected leaves.",
    "Cedar Apple Rust": "Use fungicides like Myclobutanil. Remove nearby juniper trees.",
    "Powdery Mildew": "Apply sulfur-based fungicides. Increase air circulation.",
    "Common Rust": "Use fungicides containing Azoxystrobin. Avoid overhead watering.",
    "Northern Leaf Blight": "Use chlorothalonil-based fungicides. Rotate crops.",
    "Black Rot (Grape)": "Spray with Dithane M-45 or copper-based fungicides.",
    "Huanglongbing": "No cure; remove infected trees and control citrus psyllid.",
    "Bacterial Spot": "Apply copper sprays. Avoid overhead irrigation.",
    "Early Blight": "Apply chlorothalonil-based fungicides. Avoid plant stress.",
    "Late Blight": "Use fungicides like Ridomil Gold. Remove infected plants.",
    "Septoria Leaf Spot": "Use copper fungicides. Rotate crops.",
    "Spider Mites": "Use neem oil or insecticidal soap. Increase humidity.",
    "Mosaic Virus": "No chemical treatment; remove infected plants.",
    "Yellow Leaf Curl Virus": "Control whiteflies with neem oil or insecticides."
}

# Static crop details
crop_details = {
    "Apple": "Apples require well-drained soil and regular pruning to prevent disease.",
    "Blueberry": "Blueberries prefer acidic soil and full sun for best growth.",
    "Cherry": "Cherry trees thrive in cool climates with well-aerated soil.",
    "Corn": "Corn needs plenty of sunlight and nitrogen-rich soil to grow well.",
    "Grape": "Grapes require trellises for support and regular pruning to prevent mold.",
    "Orange": "Oranges grow best in warm climates with regular watering.",
    "Peach": "Peaches need full sun and good air circulation to prevent fungal infections.",
    "Pepper": "Peppers require warm temperatures and consistent watering.",
    "Potato": "Potatoes grow well in loose, well-drained soil with proper crop rotation.",
    "Raspberry": "Raspberries need well-drained soil and proper spacing to prevent disease.",
    "Soybean": "Soybeans need moderate rainfall and well-drained soil for best yield.",
    "Squash": "Squash plants require plenty of space and good airflow to prevent mildew.",
    "Strawberry": "Strawberries thrive in well-drained, sandy soil with full sun.",
    "Tomato": "Tomatoes grow best in warm temperatures with regular fertilization."
}

# Load the model properly
MODEL_PATH = os.path.join(settings.BASE_DIR, 'plant-disease-model.pth')

#define the number of diseases classes in the trained model
num_diseases = 38

# Instantiate the model before loading weights
model = ResNet9(3, num_diseases)  # Ensure same architecture as used during training

# Load the saved state_dict if applicable
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

# Move model to CPU and set to evaluation mode
model.to('cpu')
model.eval()

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match model input size
    transforms.ToTensor(),          # Convert to tensor
])


# User Registration
def register_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]
        confirm_password = request.POST["confirm_password"]
        
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect("register")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return redirect("register")

        user = User.objects.create_user(username=username, email=email, password=password)
        user.save()
        messages.success(request, "Registration successful. You can now log in.")
        return redirect("login")

    return render(request, "register.html")

# User Login
def login_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            messages.error(request, "Invalid username or password.")

    return render(request, "index.html")

# User Logout
def logout_view(request):
    logout(request)
    return redirect("login")

# Home Page (After login)
@login_required
def home_view(request):
    detection_result = None  # Default value for detection result
    
    if request.method == 'POST' and 'image' in request.FILES:
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            detection_result = detect_disease(image)  # Call your model for disease detection
            
    else:
        form = ImageUploadForm()  # Create the form instance if not a POST request
    
    return render(request, 'home.html', {'form': form, 'detection_result': detection_result})

def detect_disease(image):
    """ Process uploaded image and predict plant disease. """
    try:
          # Save the uploaded image
        unique_filename = f"uploads/{uuid.uuid4().hex}.jpg"  # Generate unique filename
        image_path = default_storage.save(unique_filename, image)
        image_url = f"/media/{image_path}"  # URL for template rendering

        # Convert uploaded file into PIL Image
        img = Image.open(image).convert("RGB")
        img = transform(img).unsqueeze(0)  # Apply transformation and add batch dimension
        img = img.to('cpu')  # Move to CPU

        # Perform prediction
        with torch.no_grad():
            preds = model(img)
            _, predicted = torch.max(preds, dim=1)

        # Get predicted class
        predicted_class = disease_classes[predicted.item()]

        # Extract plant name and disease name
        parts = predicted_class.split()
        plant_name = parts[0]  # First word is the plant name
        disease_name = " ".join(parts[1:]) if len(parts) > 1 else "Healthy"

        # Get fertilizer recommendation if disease is present
        recommendation = fertilizer_recommendations.get(disease_name, "No fertilizer needed for healthy plants.")

        # Get crop details from static dictionary
        crop_info = crop_details.get(plant_name, "No specific details available for this plant.")

        return {
            "preview": image_url,
            "crop_name": plant_name,
            "crop_details": crop_info,
            "predicted_disease": disease_name,
            "fertilizer_recommendation": recommendation
        }

    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}


# Performance Page (Protected)
@login_required
def performance_view(request):
    return render(request, 'performance.html')

# Charts Page (Protected)
@login_required
def charts_view(request):
    return render(request, 'charts.html')

