import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QPushButton, QLabel, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import MedicalImageClassifier

class MedicalClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Image Classifier")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MedicalImageClassifier(num_classes=3, model_name='resnet50')
        self.model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create image display label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        layout.addWidget(self.image_label)
        
        # Create result label
        self.result_label = QLabel("No image selected")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)
        
        # Create buttons
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)
        
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)
        self.predict_button.setEnabled(False)
        layout.addWidget(self.predict_button)
        
        # Initialize image path
        self.image_path = None
    
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Medical Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.dcm)"
        )
        
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.predict_button.setEnabled(True)
            self.result_label.setText("Image loaded. Click 'Predict' to classify.")
    
    def predict(self):
        if not self.image_path:
            return
        
        try:
            # Load and preprocess image
            image = Image.open(self.image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            # Map prediction to label
            label_mapping = {0: "Normal", 1: "Pneumonia", 2: "Cardiomegaly"}
            result = f"Prediction: {label_mapping[prediction]}\nConfidence: {confidence:.2%}"
            self.result_label.setText(result)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = MedicalClassifierGUI()
    window.show()
    sys.exit(app.exec()) 