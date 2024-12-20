import cv2
import numpy as np
import torch.nn as nn
from ultralytics import YOLO

class ObjectDetection(nn.Module):
    def __init__(self, checkpoint_path):
        super(ObjectDetection, self).__init__()
        self.model = YOLO(checkpoint_path)
        self.names = self.model.names

        # 실제 객체의 폭 (cm)
        self.known_widths = {
            'red_light' : 8.99,
            'green_light' : 8.99,
            'cross_walk' : 3.44,    
            'stop_line' : 3.44
        }

        self.known_heights = {
            'person' : 12.12,
            'goat' : 8.59,
            'obstacle' : 10.48
        }

        # 카메라 캘리브레이션 값
        self.mtx = np.array([[672.52, 0, 330.84],
                             [0, 673.15, 257.85],
                             [0, 0, 1]])
                        
        self.dist = np.array([[-0.44717, 0.63559, 0.0029907, -0.00055208, -0.94232]])

        # 설정값
        self.KNOWN_HEIGHT = 6.2  # 사각형 높이 cm
        self.KNOWN_DISTANCE = 31.1  # 실제 거리 cm
        self.focal_length_found = self.focal_length(self.KNOWN_DISTANCE, self.KNOWN_HEIGHT, 104)

        self.prev_order = None
        self.cur_order = None

    # focal length 계산 함수
    def focal_length(self, measured_distance, real_width, width_in_rf_image):
        return (width_in_rf_image * measured_distance) / real_width

    # 거리 계산 함수
    def distance_finder(self, focal_length, real_width, width_in_frame):
        return round((real_width * focal_length) / width_in_frame - 16, 3)

    # bgr 기준
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

    # 바운딩 박스 폭 계산 함수
    def pixel_width_data(self, results):
        class_names = []  # object 이름
        widths = []  # 바운딩 박스 폭 or 높이
        boxes = []  # 바운딩 박스 좌표
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            cls_id = int(result.cls[0])
            cls_name = self.names[cls_id]

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

    # 신호 보내기
    def control_signal(self, name, distance, boxes, class_names):
        order = None
        obst_pos = None
        cls_name = None
        result_boxes = None
        if (name == 'red_light'): 
            if (distance < 33):
                order = "stop"
                cls_name = name
            else:
                if ('cross_walk' in class_names) and (distance < 50):
                    order = "stop"
                    cls_name = name          
        elif (name == 'person'):
            if distance < 17:
                order = "stop"
                cls_name = name
                result_boxes = boxes
        elif (name == 'obstacle'):
            if distance < 30:
                order = "drive"
                obst_pos = boxes
                cls_name = name
                result_boxes = boxes
        elif (name == 'goat'):
            if distance < 17:
                order = "stop"
                cls_name = name
                result_boxes = boxes
        elif (name == 'green_light'):
            if distance < 60:
                cls_name = name
        else:
            pass

        return order, obst_pos, cls_name, result_boxes
    
    def forward(self, frame):
        h, w = frame.shape[:2]
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))

        # 카메라 영상 왜곡 계수 보정
        frame = cv2.undistort(frame, self.mtx, self.dist, new_matrix)

        # YOLO 예측
        frame_results = self.model.predict(frame, conf=0.55, verbose=False)
        
        # 바운딩 박스 폭 or 높이 구하기
        class_names, widths, boxes = self.pixel_width_data(frame_results)

        # 현재 프레임에 대한 order 값
        order_list = []

        # 현재 프레임에서 인식된 object
        cls_list = []

        # obstacle 바운딩 박스 좌표
        obst = None

        for name, width, cls_boxes in zip(class_names, widths, boxes):
            if name in self.known_widths:  # 초록불, 빨간불
                distance = self.distance_finder(self.focal_length_found, 
                                                self.known_widths[name], width)
            elif name in self.known_heights:  # 나머지
                distance = self.distance_finder(self.focal_length_found, 
                                                self.known_heights[name], width)
            else:
                continue

            order, obst_pos, obj_name, obj_boxes = self.control_signal(name, distance, cls_boxes, class_names)

            if (obj_name != None and obj_boxes != None):
                cv2.rectangle(frame, obj_boxes[:2], obj_boxes[2:], self.color_finder(obj_name), 2)
                cv2.putText(frame, obj_name, (obj_boxes[0], obj_boxes[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_finder(obj_name), 2)
                cv2.putText(frame, f"{distance} cm", (obj_boxes[0], obj_boxes[-1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.57, self.color_finder(obj_name), 2)

            if obst_pos != None:
                obst = obst_pos

            order_list.append(order)

            if obj_name != None:
                cls_list.append(obj_name)

        if 'stop' in order_list:
            self.cur_order = 'stop'
            text_color = (34, 34, 178)
        else:
            self.cur_order = 'drive'
            text_color = (34, 139, 34)

        cv2.putText(frame, self.cur_order, (40, 400), cv2.FONT_HERSHEY_DUPLEX, 1.5, text_color, 4)

        # esp 로 명령 보내기
        if self.prev_order != self.cur_order:
            # print(self.cur_order)
            self.prev_order = self.cur_order    
        
        # 중복 제거
        cls_list = list(set(cls_list))

        self.data = {'order':self.cur_order, 'obstacle':obst, 'cls_list':cls_list}

        return self.data, frame
