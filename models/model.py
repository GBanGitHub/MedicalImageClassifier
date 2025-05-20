import torch
import torch.nn as nn
import torchvision.models as models

class MedicalImageClassifier(nn.Module):
    def __init__(self, num_classes=3, model_name='resnet50', pretrained=True):
        super(MedicalImageClassifier, self).__init__()
        
        # Load pretrained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            # Modify the final layer for our number of classes
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        elif model_name == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self(x)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities 