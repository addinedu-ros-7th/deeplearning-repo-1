import os
import timm
import torch
import torch.nn as nn
import numpy as np

from huggingface_hub import hf_hub_download
from torchvision import  transforms
from supervision import Detections
from ultralytics import YOLO

class DrowseDetection(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.face_detect = DetectionModel()
        self.drowsy_detect = DrowseDetectionModel()
        self.drowsy_detect.get_state_dict(checkpoint_path)

    def forward(self, frame):
        x1, y1, x2, y2 = self.face_detect(frame)
        drowsy = self.drowsy_detect(frame[y1:y2, x1:x2])
        return drowsy
    
class DrowseDetectionModel(nn.Module):
    def __init__(self):        
        super().__init__()
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=2)
        self.set_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
        ])

    def set_model(self):
        self.model.eval()

    def get_state_dict(self, checkpoint_path):
        path = os.path.join(checkpoint_path)
        self.model.load_state_dict(torch.load(path))

    def forward(self, x):
        x = self.transform(x)#.cuda()
        x = x.unsqueeze(0)
        x = self.model(x)
        x = x.argmax(1).item()
        return x
    
class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        self.FaceDetection = YOLO(model_path)

    def forward(self, x):
        output = self.FaceDetection(x, verbose=False)
        results = Detections.from_ultralytics(output[0])
        bbox = results.xyxy[0].astype(int) + np.array([-40, -60, 40, 10])
        return bbox