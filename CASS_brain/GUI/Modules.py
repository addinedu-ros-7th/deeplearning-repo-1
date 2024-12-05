import os
import time
import timm
import face_recognition
import serial
import torch
import torch.nn as nn
import numpy as np
import socket
from torchvision import  transforms
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections

class Camera(QThread):
    update = pyqtSignal()

    def __init__(self, sec=0, parent=None):
        super().__init__()
        self.main = parent
        self.running = True

    def run(self):
        while self.isRunning:
            self.update.emit()
            time.sleep(0.07)

    def stop(self):
        self.running = False

class Arduino(QThread):
    distance_signal = pyqtSignal(str)  # 거리를 전달할 시그널

    def __init__(self, parent=None):
        super().__init__()
        self.main = parent
        self.client_socket = None
        self.esp32_ip = '192.168.199.119'  # ESP32의 IP 주소
        self.esp32_port = 8080  # ESP32에서 설정한 포트

    def run(self):
        # 아두이노 시리얼 포트 열기
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
            self.client_socket.connect((self.esp32_ip, self.esp32_port))
            time.sleep(1)  # 시리얼 연결 대기
            print('ESP32 Connected')
        except serial.SerialException as e:
            print(f"Could not open serial port: {e}")
 
        while True:
            try:
                self.client_socket.settimeout(0.01) 
                data = self.client_socket.recv(1024).decode()
                self.distance_signal.emit(data)  # 시그널로 데이터 전달
                time.sleep(0.03)
            except:
                pass

    def stop(self):
        print('ESP32 Disconnected')
        self.client_socket.close()  # 시리얼 포트 닫기


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

    def face_athentication(self, frame):
        face_locations = face_recognition.face_locations(frame, model="cnn")
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.35)
            name = "unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            face_names.append(name)
            if len(face_locations) == 0:
                face_locations = 0

        return face_locations, face_names