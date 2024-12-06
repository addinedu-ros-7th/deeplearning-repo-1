import face_recognition  
import torch
import torch.nn as nn
from torchvision import transforms
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import timm
import cv2
import os
import numpy as np


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

class FaceRecognitionModel():
    def __init__(self):
        super().__init__()
        self.known_face_encodings = []
        self.known_face_names = []

    def set_user_image(self, path):
        self.my_image = face_recognition.load_image_file(path)
        self.my_face_encoding = face_recognition.face_encodings(self.my_image)[0]

    def set_known_user(self, encoding, name):
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(name)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

face_recognition_model = FaceRecognitionModel()
path = "./test/data/face/my_img/soyoung.png"
name = "soyoung"

face_recognition_model.set_user_image(path)
face_recognition_model.set_known_user(face_recognition_model.my_face_encoding, "soyoung")

while True:
    ret, frame = video_capture.read()

    if frame is not None:
        face_locations = face_recognition.face_locations(frame, model="cnn")
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(face_recognition_model.known_face_encodings, face_encoding, tolerance=0.35)
            print(matches)
            name = "unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = face_recognition_model.known_face_names[first_match_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left - 10, top - 10), (right + 10, bottom + 10), (200, 100, 5), 2)
            cv2.rectangle(frame, (left - 10, bottom - 25), (right + 10, bottom + 10), (200, 100, 5), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, .5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("not connected to video")

        try:
            ret, frame = video_capture.read()
        except:
            break

class DrowseDetectionModel(nn.Module):
    def __init__(self):        
        super().__init__()
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=2)
        # self.model = models.resnet50(weights='IMAGENET1K_V2')
        self.set_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
        ])

    def set_model(self):
        self.model.eval()


    def get_state_dict(self, checkpoint_path):
        path = os.path.join(checkpoint_path, 'checkpoints/best_model.pth')
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

video_capture.release()
cv2.destroyAllWindows()