import os
import cv2
import time
import timm
import face_recognition
import serial
import torch
import torch.nn as nn
import numpy as np
import socket
import mysql.connector
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

class TCP():
    def __init__(self):
        self.client_socket = None
        self.esp32_ip = '192.168.9.46'  # ESP32의 IP 주소
        self.esp32_port = 8080 
        self.message = None
        self.isConnected = False

    def connectLeg(self):
        print("try connecting to CASS_leg...")
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        if self.isConnected == False:
            try:
                self.client_socket.connect((self.esp32_ip, self.esp32_port))
                time.sleep(0.5)
                response = self.sendMsg("connect")
                if type(response) == bytes:
                    response = response.decode('utf-8')
                    print(response)
                if response == "connected":
                    self.isConnected = True
                    print("="*10, "leg has well connected", "="*10)
                else:
                    print("not byte not str what are you", response)

            except(ConnectionRefusedError):
                print("connection refused")
                self.isConnected = False
                self.connectLeg()
        else:
            print("CASS_leg is already connected")

    def sendMsg(self, msg):
        match(msg):
            case "connect":
                self.message = "11"
            case "soyoung": # user_name
                self.message = "21"
            case "drive":
                self.message = "31"
            case "stop":
                self.message = "32"
            case "reverse":
                self.message = "33"
            case "L1":
                self.message = "41"
            case "L2":
                self.message = "42"
            case "L3":
                self.message = "43"
            case "R1":
                self.message = "51"
            case "R2":
                self.message = "52"
            case "R3":
                self.message = "53"
            case "emergency":
                self.message = "99"
        
        """
        for test without tcp
        do annotate send, recv
        undo annotate "test"
        """
        # self.client_socket.settimeout(0.02) 

        # self.message = self.message.encode()
        # self.client_socket.send(self.message) # send
        # print("CASS_brain said : ", msg)
        # response = self.client_socket.recv(1024) #recv
        # response = response.decode('utf-8') 
        # response = "test" # for test
        # print("CASS_leg said : ", response)
        # # print("=" * 40)

        # return response
    
    def close(self):
        self.client_socket.close()

class DataBase():
    def __init__(self):
        self.local = None

    def connectLocal(self):
        self.local = mysql.connector.connect(
            host = "localhost",
            user = "root",
            password = "0050",
            database = "CASS"
        )

    def fetchUserData(self, path):
        self.cur = self.local.cursor(buffered=True)
        self.cur.execute(f"SELECT * WHERE path = {path} FROM DRIVER")
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
        print(sql_id)
        self.cur.execute(sql_id)
        driver_id = self.cur.fetchone()[0]
        print("registere success :", driver_id)

        return driver_id

    def registerPhoto(self, id, path):
        self.cur = self.local.cursor(buffered=True)
        sql = "UPDATE DRIVER SET path = %s WHERE id = %s"
        val = (path, id)
        self.cur.execute(sql, val)
        self.local.commit()

        print("registere success :", path)

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

class Arduino(QThread):
    distance_signal = pyqtSignal(str)  # 거리를 전달할 시그널

    def __init__(self, parent=None):
        super().__init__()
        self.main = parent
        self.client_socket = None
        self.esp32_ip = '172.20.10.10'  # ESP32의 IP 주소
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
        path = os.path.join(checkpoint_path, 'best_model.pth')
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

    def draw_boxes(self, frame, face_locations, face_names):
        for (top, right, bottom, left), name in zip(face_locations, face_names):
                    cv2.rectangle(frame, (left - 10, top - 10), (right + 10, bottom + 10), (200, 100, 5), 2)
                    cv2.rectangle(frame, (left - 10, bottom - 25), (right + 10, bottom + 10), (200, 100, 5), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, .5, (255, 255, 255), 1)

class Centroid():
    def __init__(self):
        self.centroid_x, self.centroid_y = 0, 0

    def get_centroid(self, polygon):
        area = 0
        self.centroid_x = 0
        self.centroid_y = 0
        n = len(polygon)

        for i in range(n):
            j = (i + 1) % n
            factor = polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
            area += factor
            self.centroid_x += (polygon[i][0] + polygon[j][0]) * factor
            self.centroid_y += (polygon[i][1] + polygon[j][1]) * factor
        area /= 2.0
        if area != 0:
            self.centroid_x /= (6 * area)
            self.centroid_y /= (6 * area)
