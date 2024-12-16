import os
import cv2
import time
import timm
import torch
import torch.nn as nn
import numpy as np
import mysql.connector
import face_recognition

from PyQt5.QtCore import QThread, pyqtSignal
from huggingface_hub import hf_hub_download
from torchvision import  transforms
from supervision import Detections
from ultralytics import YOLO

class Thread(QThread):
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

class TCP():
    def __init__(self):
        self.esp32_ip = '192.168.9.46'  # ESP32의 IP 주소
        self.esp32_port = 8080 
        self.message = None

    def encodeMsg(self, msg):
        match(msg):
            case "connect":
                self.message = "11\n"
            case "soyoung": # user_name
                self.message = "21\n"
            case "drive":
                self.message = "31\n"
            case "stop":
                self.message = "32\n"
            case "reverse":
                self.message = "33\n"
            case "accel":
                self.message = "34\n"
                print("accel sent")
        
            case "L1":
                self.message = "41\n"
            case "L2":
                self.message = "42\n"
            case "L3":
                self.message = "43\n"
            case "R1":
                self.message = "51\n"
            case "R2":
                self.message = "52\n"
            case "R3":
                self.message = "53\n"
            case "side_parking":
                self.message = "88\n"
            case "emergency":
                self.message = "99\n"
        
        return self.message

class DataBase():
    def __init__(self):
        self.local = None

    def connectLocal(self, password):
        self.local = mysql.connector.connect(
            host = "localhost",
            user = "root",
            password = password,
            database = "CASS"
        )
    def closeLocal(self):
        self.local.close()

    def fetchUserData(self, path):
        self.cur = self.local.cursor(buffered=True)
        sql = "SELECT * FROM DRIVER WHERE path = %s"
        self.cur.execute(sql, (path))
        data = self.cur.fetchall()
        driver_id = data[0][0]
        driver_name = data[0][1]
        driver_birth = data[0][2]
        driver_contact = data[0][3]

        return driver_id, driver_name, driver_birth, driver_contact
    
    def registerUser(self, name, birth, contact):
        self.cur = self.local.cursor(buffered=True)
        sql = "INSERT INTO DRIVER (name, birth, contact) VALUES (%s, %s, %s)"
        val = (name, birth, contact)
        self.cur.execute(sql, val)
        self.local.commit()
        sql_id = "select id from DRIVER order by id desc limit 1"
        self.cur.execute(sql_id)
        driver_id = self.cur.fetchone()[0]
        print("registere success, driver id :", driver_id)

        return driver_id

    def registerPhoto(self, id, path):
        self.cur = self.local.cursor(buffered=True)
        sql = "UPDATE DRIVER SET path = %s WHERE id = %s"
        val = (path, id)
        self.cur.execute(sql, val)
        self.local.commit()

        print("registere success, photo path :", path)

class Driver():
    def __init__(self):
        self.id = None
        self.name = None
        self.birth = None
        self.contact = None

    def getInfo(self, name, birth, contact):
        self.name = name
        self.birth = birth
        self.contact = contact

        print(self.name, self.birth, self.contact)

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
    
class DrowsyDetection(nn.Module):
    def __init__(self):
        super().__init__()
        self.face_detect = DetectionModel()
        self.drowsy_detect = DrowseDetectionModel()
        self.drowsy_detect.get_state_dict('bestFace.pth')

    def forward(self, frame):
        x1, y1, x2, y2 = self.face_detect(frame)
        drowsy = self.drowsy_detect(frame[y1:y2, x1:x2])
        return drowsy
    
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

    def draw_boxes(self, frame, face_locations, face_names):
        for (top, right, bottom, left), name in zip(face_locations, face_names):
                    cv2.rectangle(frame, (left - 10, top - 10), (right + 10, bottom + 10), (200, 100, 5), 2)
                    cv2.rectangle(frame, (left - 10, bottom - 25), (right + 10, bottom + 10), (200, 100, 5), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, .5, (255, 255, 255), 1)
