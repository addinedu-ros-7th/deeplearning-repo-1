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
        self.esp32_ip = '172.20.10.10'  # ESP32의 IP 주소
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

        self.message = self.message.encode()
        self.client_socket.send(self.message) # send
        print("CASS_brain said : ", msg)
        time.sleep(0.5)
        response = self.client_socket.recv(1024) #recv
        response = response.decode('utf-8') 
        # response = "test" # for test
        print("CASS_leg said : ", response)
        print("=" * 40)

        return response
    
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

    def draw_boxes(self, frame, face_locations, face_names):
        for (top, right, bottom, left), name in zip(face_locations, face_names):
                    cv2.rectangle(frame, (left - 10, top - 10), (right + 10, bottom + 10), (200, 100, 5), 2)
                    cv2.rectangle(frame, (left - 10, bottom - 25), (right + 10, bottom + 10), (200, 100, 5), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, .5, (255, 255, 255), 1)
              
class ObjectDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        model_path = '../../../test/data/face/best.pt'
        self.model = YOLO(model_path)
        self.names = self.model.model.names

        self.known_widths = {
                                'person' : 5.54,
                                'red_light' : 8.99,
                                'green_light' : 8.99,
                                'goat' : 9.89
                            }

        self.known_heights = {
                                'person' : 12.12,
                                'goat' : 8.59,
                                'obstacle' : 10.48
                            }
        
        self.mtx = np.array([[     672.52,           0,      330.84],
                            [          0,      673.15,      257.85],
                            [          0,           0,           1]])
        
        self.dist = np.array([[   -0.44717,     0.63559,   0.0029907, -0.00055208,    -0.94232]])

        self.KNOWN_WIDTH = 6.2  # cm, 피규어 실제 너비
        self.KNOWN_DISTANCE = 31.1  # cm, 참조 거리

    def focal_length(self, measured_distance, real_width, width_in_rf_image):
        return (width_in_rf_image * measured_distance) / real_width

# 거리 계산 함수
    def distance_finder(self, focal_length, real_width, width_in_frame):
        return (real_width * focal_length) / width_in_frame

    def color_finder(self, name):
        
        if name == 'red_light':
            color = (0, 0, 255)
        elif name == 'green_light':
            color = (0, 255, 100)
        elif name == 'person':
            color = (255, 0, 200)
        elif name == 'goat':
            color = (0, 215, 255)
        elif name == 'cross_walk':
            color = (240, 240, 240)
        elif name == 'stop_line':
            color = (128, 0, 0)
        else:
            color = (100, 120, 200)
        return color

    def objectDetection(self, results, frame):
        class_names = []  # object 이름
        widths = []  # 바운딩 박스 폭 or 높이
        boxes = []  # 바운딩 박스 좌표
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            confidence = result.conf[0]
            cls_id = int(result.cls[0])
            cls_name = self.names[cls_id]
            label = f"{cls_name}: {confidence:.2f}"

            # 바운딩 박스 및 레이블 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color_finder(cls_name), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_finder(cls_name), 2)
        
            # 'person', 'goat', 'obstacle' 이면 높이를 사용
            if cls_name in self.known_heights:
                ref_image_width = y2 - y1
            # 신호등은 폭을 사용
            else:
                ref_image_width = x2 - x1

            class_names.append(cls_name)  # 클래스 이름 저장
            widths.append(ref_image_width)  # 폭 or 높이 저장
            boxes.append((x1, y1, x2, y2))  # 바운딩 박스 좌표 저장
        return class_names, widths, boxes 

    def get_distance(self, frame):
            h, w = frame.shape[:2]
            focal_length_found = 520.925
            new_matrix, _ = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))

            # 왜곡 계수 보정
            calibrated_frame = cv2.undistort(frame, self.mtx, self.dist, new_matrix)

            # YOLO 예측
            frame_results = self.model.predict(calibrated_frame, conf=0.55, verbose=False)
                
            class_names, widths, boxes = self.objectDetection(frame_results, calibrated_frame)
            detect_list = []

            for name, width, (x1, y1, x2, y2) in zip(class_names, widths, boxes):
                if name in self.known_widths:  # 초록불, 빨간불
                    distance = self.distance_finder(focal_length_found, self.known_widths[name], width) - 16
                elif name in self.known_heights:
                    distance = self.distance_finder(focal_length_found, self.known_heights[name], width) - 16
                else:
                    continue

                cv2.putText(calibrated_frame, f"{name} : {round(distance, 2)} cm", 
                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_finder(name), 2)
                detect_list.extend([name, distance])

            return detect_list
    
    def control_signal(self, name, distance, class_names):
        order = ""
        # 멈춰야 하는 경우
        if (name == 'red_light'): 
            if (distance < 33):
                order = "Stop"
            else:
                if 'cross_walk' in class_names:
                    order = "Stop"
        elif (name == 'person'):
            if distance < 12:
                order = "Stop"
        elif (name == 'obstacle'):
            if distance < 15:
                order = "Avoidance"
        elif (name == 'goat'):  # goat
            if distance < 12:
                order = "Stop"

        return order

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


class LaneSegmentation(nn.Module):
    def __init__(self, checkpoint_path):
        super(LaneSegmentation, self).__init__()
        self.model = YOLO(checkpoint_path)
        self.center = Centroid()
        self.N2C = {'0': 'center_road', '1': 'left_road', '2': 'right_road'}

    def forward(self, x, direction='stright'):
        output = self.model(frame, verbose=False, device=0)
        frame = output[0].orig_img

        cls = output[0].boxes.cls
        img_size = output[0].masks.orig_shape

        mid_ = int(img_size[1]/2)
        bot_ = img_size[0]

        mask = torch.zeros((img_size[0], img_size[1], 3)).cuda()
        for i, data in enumerate(output[0].masks.data):
            if 'center_road' in self.N2C[cls[i].item()]:
                mask[:,:,0][data==1] = 100
            elif 'left_road' in self.N2C[cls[i].item()]:
                mask[:,:,1][data==1] = 100
            elif 'right_road' in self.N2C[cls[i].item()]:
                mask[:,:,2][data==1] = 100
                
        mask = mask.detach().cpu()

        roads = {'left':[], 'right':[], 'center':[]}
        for i, xy in enumerate(output[0].masks.xy):
            self.center.get_centroid(xy)
            point = np.array([self.center.centroid_x, center.centroid_y], dtype=np.int32)
            c_name = self.N2C[cls[i].item()]
            if c_name=='center_road':
                roads['center'].append(np.expand_dims(point, axis=0))
            elif c_name=='left_road':
                roads['left'].append(np.expand_dims(point, axis=0))
            elif c_name=='right_road':
                roads['right'].append(np.expand_dims(point, axis=0))
        

        start_point = (mid_, bot_)
        left = roads['left']
        right = roads['right']
        center = roads['center']

        is_left = len(left) != 0
        is_right = len(right) != 0
        is_center = len(center) != 0

        if is_left:
            left = np.concatenate(left, axis=0).mean(0).astype(np.int32)
        if is_right:
            right = np.concatenate(right, axis=0).mean(0).astype(np.int32)
        if is_center:
            center = np.concatenate(center, axis=0)
            center = center[center[:, 0].argsort()]

        if direction == 'left':
            center = center[0]
        elif direction == 'right':
            center = center[-1]
        else:
            if len(center) > 2:
                center = center[1:-1].mean(0).astype(np.int32)
            else:
                center = center[0]        

        cv2.arrowedLine(frame, start_point, center,
                            color=(0, 0, 0), 
                            thickness=5, tipLength=0.1)
        slope = -(start_point[0] - center[0])/(start_point[1] - center[1])
        slope = np.rad2deg(np.arctan(slope)) 
        
        return slope