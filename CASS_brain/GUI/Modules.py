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
        self.esp32_ip = '172.20.10.8'  # ESP32의 IP 주소
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
        

        self.message = self.message.encode()
        self.client_socket.send(self.message)
        print("CASS_brain said : ", msg)
        time.sleep(0.5)
        response = self.client_socket.recv(1024)
        response = response.decode('utf-8')
        # response = "test"
        print("CASS_leg said : ", response)
        print("=" * 40)

        return response
    
    def close(self):
        self.client_socket.close()


class Arduino(QThread):
    distance_signal = pyqtSignal(str)  # 거리를 전달할 시그널

    def __init__(self, parent=None):
        super().__init__()
        self.main = parent
        self.client_socket = None
        self.esp32_ip = '172.20.10.8'  # ESP32의 IP 주소
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

        self.KNOWN_WIDTH = 5.54  # cm, 피규어 실제 너비
        self.KNOWN_DISTANCE = 29.9  # cm, 참조 거리

    def focal_length(self, measured_distance, real_width, width_in_rf_image):
        return (width_in_rf_image * measured_distance) / real_width

# 거리 계산 함수
    def distance_finder(self, focal_length, real_width, width_in_frame):
        return (real_width * focal_length) / width_in_frame

    def color_finder(self, name:str):
        if name == 'red_light':
            color = (0, 0, 255)
        elif name == 'green_light':
            color = (0, 255, 0)
        elif name == 'person':
            color = (255, 0, 200)
        elif name == 'goat':
            color = (0, 215, 255)
        else:
            color = (100, 120, 200)
        return color

    def objectDetection(self, results, frame):
        class_names = []
        widths = []
        boxes = []  

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            confidence = result.conf[0]
            class_id = int(result.cls[0])
            obj = self.names[class_id]
            label = f"{obj}: {confidence:.2f}"
            color_class = self.color_finder(obj)

            cv2.rectangle(frame, (x1, y1), (x2, y2), 
                          color_class, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                        color_class, 2)

            ref_image_width = x2 - x1
            class_names.append(obj)
            widths.append(ref_image_width)
            boxes.append((x1, y1, x2, y2))

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

class LaneSegmentation(nn.Module):
    def __init__(self):
        super(LaneSegmentation, self).__init__()
        model_path = "../../../test/data/face/lane_best.pt"
        self.model = YOLO(model_path)

    def forward(self, x, direction='stright'):
        try:
            output = self.model(x, verbose=False, device=0)
            frame = output[0].orig_img

            cls = output[0].boxes.cls
            img_size = output[0].masks.orig_shape
            mask = torch.zeros((img_size[0], img_size[1], 3)).cuda()
            for i, data in enumerate(output[0].masks.data):
                if cls[i]%2 == 0:
                    mask[:,:,0][data==1] = 100
                else:
                    mask[:,:,1][data==1] = 100
            mask = mask.detach().cpu()
            # cv2.imshow('mask', mask.numpy())
            
            mid_ = int(img_size[1]/2)
            bot_ = img_size[0]
            start_point = (mid_, bot_)

            pos1 = {
                'curve'   : {'left':[], 'right':[],},
                'stright' : {'left':[], 'right':[],}
            }
            pos2 = {'left':[], 'right':[]}
            const = 300
            for idx, xy in enumerate(output[0].masks.xy):
                xy[:,1][xy[:,1]<const]= 0
                xy[:,0][xy[:,1]==0] = 0
                l = len(xy[:,1][xy[:,1]>=const])
                y_pts = xy[:,1][xy[:,1]!=0]
                if len(y_pts) == 0:
                    continue
                point = (xy.sum(0)/l).astype(np.int32)
                y_max = y_pts.max()
                x_max = xy[:,0][xy[:,1]==y_max].mean()
                class_ = cls[idx].item()
                
                x_diff = x_max - mid_
                b_point = (x_max, y_max)
                slope = abs((y_max - point[1])/(x_max - point[0] + 1e-6))
                x_diff2 = b_point[0] - point[0]
                point = np.concatenate([point, [slope]], axis=0) #point, slope, class_
                lane_info = (point, b_point, x_diff2)

                if cls[idx]%2 == 0:
                    if class_ >= 4:
                        pos2['left'].append(np.expand_dims(point, axis=0))
                    else:
                        if x_diff < 0:
                            pos1['curve']['left'].append(lane_info)
                        else:
                            pos1['curve']['right'].append(lane_info)
                else:
                    if class_ >= 4:
                        pos2['left'].append(np.expand_dims(point, axis=0))
                    else:
                        if x_diff < 0:
                            pos1['stright']['left'].append(lane_info)
                        else:
                            pos1['stright']['right'].append(lane_info)
            cnst=0.25
            if direction == 'stright':
                LeftRight = pos1[direction]
                L, R = LeftRight.values()
                if not(len(L)==0 and len(R)==0):

                    for key, vals in LeftRight.items():
                        for val in vals:
                            points, b_point, x_diff2 = val
                            point = points[:2]
                            slope = points[-1]
                            if key=='left':
                                if slope > cnst and x_diff2 < 0:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
                                else:
                                    pos2['right'].append(np.expand_dims(points, axis=0))
                            elif key=='right':
                                if slope > cnst and x_diff2 > 0:
                                    pos2['right'].append(np.expand_dims(points, axis=0))
                                else:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
                else:
                    LeftRight = pos1['curve']
                    for key, vals in LeftRight.items():
                        for val in vals:
                            points, b_point, x_diff2 = val
                            point = points[:2]
                            slope = points[-1]
                            if key=='left':
                                if slope > cnst and x_diff2 < 0:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
                                else:
                                    pos2['right'].append(np.expand_dims(points, axis=0))
                            elif key=='right':
                                if slope > cnst and x_diff2 > 0:
                                    pos2['right'].append(np.expand_dims(points, axis=0))
                                else:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
            else:
                LeftRight = pos1[direction]
                L, R = LeftRight.values()

                if not(len(L)==0 and len(R)==0):

                    for key, vals in LeftRight.items():
                        for val in vals:
                            points, b_point, x_diff2 = val
                            point = points[:2]
                            slope = points[-1]
                            if key=='left':
                                if slope > cnst and x_diff2 < 0:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
                                else:
                                    pos2['right'].append(np.expand_dims(points, axis=0))
                            elif key=='right':
                                if slope > cnst and x_diff2 > 0:
                                    pos2['right'].append(np.expand_dims(points, axis=0))
                                else:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
                else:
                    LeftRight = pos1['stright']
                    for key, vals in LeftRight.items():
                        for val in vals:
                            points, b_point, x_diff2 = val
                            point = points[:2]
                            slope = points[-1]
                            if key=='left':
                                if slope > cnst and x_diff2 < 0:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
                                else:
                                    pos2['right'].append(np.expand_dims(points, axis=0))
                            elif key=='right':
                                if slope > cnst and x_diff2 > 0:
                                    pos2['right'].append(np.expand_dims(points, axis=0))               
                                else:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
            cnst_s1 = 160
            cnst_s2 = 270
            # 빈칸 채우기
            if len(pos2['left'])==0:
                rpoint = np.concatenate(pos2['right'], axis=0)
                rpoint = rpoint[rpoint[:, 1].argmax()][:2]
                diff = rpoint[0] - mid_
                if direction=='stright':
                    if diff < cnst_s1:
                        pos2['left'].append(np.array([[5, rpoint[1]]]))
                    elif diff > cnst_s2:
                        pos2['left'].append(np.array([[mid_, rpoint[1]]]))
                    else:
                        pos2['left'].append(np.array([[img_size[1]-rpoint[0], rpoint[1]+10]]))
                elif direction=='curve':
                    if diff < cnst_s1:
                        pos2['left'].append(np.array([[5, rpoint[1]]]))
                    elif diff > cnst_s2:
                        pos2['left'].append(np.array([[mid_, rpoint[1]]]))
                    else:
                        pos2['left'].append(np.array([[img_size[1]-rpoint[0], rpoint[1]]]))

            elif len(pos2['right'])==0:
                rpoint = np.concatenate(pos2['left'], axis=0)
                rpoint = rpoint[rpoint[:, 1].argmax()][:2]
                diff = mid_ - rpoint[0]
                if direction=='stright':
                    if diff < cnst_s1:
                        pos2['right'].append(np.array([[img_size[1]-5, rpoint[1]]]))
                    elif diff > cnst_s2:
                        pos2['right'].append(np.array([[mid_, rpoint[1]]]))
                    else:
                        pos2['right'].append(np.array([[img_size[1]-rpoint[0], rpoint[1]+10]]))
                elif direction=='curve':

                    if diff < cnst_s1:
                        pos2['right'].append(np.array([[img_size[1]-5, rpoint[1]]]))
                    elif diff > cnst_s2:
                        pos2['right'].append(np.array([[mid_, rpoint[1]]]))
                    else:
                        pos2['right'].append(np.array([[img_size[1]-rpoint[0], rpoint[1]]]))
                
                
            left = np.concatenate(pos2['left'], axis=0)
            right = np.concatenate(pos2['right'], axis=0)
            left = left[left[:, 1].argmax()][:2]
            right = right[right[:, 1].argmax()][:2]
            cv2.circle(frame, (int(left[0]), int(left[1])), 10, (0, 0, 255), -1)
            cv2.circle(frame, (int(right[0]), int(right[1])), 10, (0, 0, 255), -1)
            center = ((left + right)/2).astype(np.int32)
            diff_x = center[0] - mid_

            cv2.arrowedLine(frame, start_point, center,
                                color=(0, 0, 0), 
                                thickness=5, tipLength=0.1)
            # 방향 return 
            return diff_x
        except:
            pass