from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
from Modules import *
import cv2
import sys
import socket

form_class = uic.loadUiType("CASS_ui.ui")[0]

ESP32_IP = "172.20.10.8"
ESP32_PORT = 8080
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

class MainWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setCamThreads()

        self.duration = 0
        self.athentified = False

        self.ath_model = FaceRecognitionModel() # face athentification model
        self.setUserImage()

        self.detect_model = ObjectDetectionModel()
    
        self.setLabelCams()
        self.cameraOn()

        #for testing
        self.btnSend.clicked.connect(self.sendData)
        self.lineEdit.returnPressed.connect(self.sendData)

    def setCamThreads(self):
        self.faceCam = Camera()
        self.daemon = True
        self.faceCam.update.connect(self.updateFaceCam)

        self.legCam = Camera()
        self.daemon = True
        self.legCam.update.connect(self.updateLegCam)
    
    def setLabelCams(self):
        self.width, self.height = self.labelFace.width(), self.labelFace.height()
        self.facePixmap = QPixmap(self.width, self.height)

        self.w, self.h = self.labelLegCam.width(), self.labelLegCam.height()
        self.legPixmap = QPixmap(self.w, self.h)

    def msgMaker(self):
        self.msg_list = []

    def addMsg(self, msg):
        self.msg_list.append(msg)

    def sendData(self):
        message = self.lineEdit.text()
        '''
        TODO: need to solve error triggerd by "\msg\n"
        '''
        print(message)
        msg = message.encode()
        # client_socket.send(msg)
        print(msg)
        # print("CASS_brain said : ", msg)
        # response = client_socket.recv(1024)
        # print("CASS_leg said : ", response)

    def setUserImage(self):
        '''
        TODO:needkk to create user image
        '''
        self.path = "../../../test/data/face/my_img/soyoung.png"
        self.name = "soyoung"

        self.ath_model.set_user_image(self.path)
        self.ath_model.set_known_user(self.ath_model.my_face_encoding, self.name)

    def cameraOn(self):
        '''
        turn a camera on when driver authentification is success
        '''
        if not self.athentified:
            self.start = time.time()
            self.faceCam.start()
            self.faceCam.isRunning = True
            self.faceVideo = cv2.VideoCapture(0)
            print("camera on")

        elif self.athentified:
            self.labelFace.hide()
            print("authentification success")
            self.legCam.start()
            self.legCam.isRunning = True
            self.legVideo = cv2.VideoCapture(2)

    def legCamOn(self):
        '''
        in case not using face_recognition
        '''
        self.labelFace.hide()
        self.legCam.start()
        self.legCam.isRunning = True
        self.legVideo = cv2.VideoCapture(2)

    def authentification(self, frame):
        '''
        authentify a driver face duration 3 secs
        '''
        self.face_locations, self.face_names = self.ath_model.face_athentication(frame)

        if self.face_names[0] == self.name:
            if self.athentified == False:
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    cv2.rectangle(frame, (left - 10, top - 10), (right + 10, bottom + 10), (200, 100, 5), 2)
                    cv2.rectangle(frame, (left - 10, bottom - 25), (right + 10, bottom + 10), (200, 100, 5), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, .5, (255, 255, 255), 1)
            
                if self.duration > 3 and self.duration < 4:
                    self.athentified = True
                    QMessageBox.warning(self, "Authentification ", f"{self.name} Driver Authentification Success.")
                    message = "connect"
                    msg = message.encode()
                    client_socket.connect((ESP32_IP, ESP32_PORT))
                    client_socket.send(msg)
                    print("CASS_brain said : ", msg)
                    response = client_socket.recv(1024) 
                    print("CASS_leg said : ", response)
                    self.labelFace.hide()

    def updateFaceCam(self):
        ret, face_frame = self.faceVideo.read()
        
        if ret:
            frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            
            self.end = time.time()
            self.duration = self.end - self.start

            self.authentification(frame)
            
            # self.DrowsyDetection(frame)
            
            # if self.isDrowsy1 != self.isDrowsy2:
                # self.send_Drowsy()
                
            h, w, c = frame.shape
            qImg = QImage(frame, w, h, w*c, QImage.Format_RGB888)
            self.facePixmap = self.facePixmap.fromImage(qImg)
            self.facePixmap = self.facePixmap.scaled(self.width, self.height)
            self.labelFace.setPixmap(self.facePixmap)

    def objectDetection(self, results, frame):
        class_names = []
        widths = []
        boxes = []  
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            confidence = result.conf[0]
            class_id = int(result.cls[0])
            label = f"{self.detect_model.names[class_id]}: {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), self.detect_model.color_finder(self.detect_model.names[class_id]), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.detect_model.color_finder(self.detect_model.names[class_id]), 2)

            ref_image_width = x2 - x1
            class_names.append(self.detect_model.names[class_id])
            widths.append(ref_image_width)
            boxes.append((x1, y1, x2, y2))

        return class_names, widths, boxes

    def updateLegCam(self):
        ret, self.frame = self.legVideo.read()
        
        if ret:
            # frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            frame = self.frame.copy()
            h, w, c = frame.shape

            new_matrix, _ = cv2.getOptimalNewCameraMatrix(self.detect_model.mtx, self.detect_model.dist, (w, h), 1, (w, h))
            calibrated_frame = cv2.undistort(frame, self.detect_model.mtx, self.detect_model.dist, new_matrix)
            frame_results = self.detect_model.model.predict(calibrated_frame, conf=0.55, verbose=False)
            focal_length_found = 5.
            '''TODO: need to change focal_length_found data'''

            class_names, widths, boxes = self.detect_model.pixel_width_data(frame_results, calibrated_frame)

            for name, width, (x1, y1, x2, y2) in zip(class_names, widths, boxes):
                if name in self.detect_model.known_widths:
                    distance = self.detect_model.distance_finder(focal_length_found, self.detect_model.known_widths[name], width) - 16
                    cv2.putText(calibrated_frame, f"{round(distance, 2)} cm", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.detect_model.color_finder(name), 2)

            qImg = QImage(calibrated_frame, w, h, w*c, QImage.Format_RGB888)
            self.legPixmap = self.legPixmap.fromImage(qImg)
            self.legPixmap = self.legPixmap.scaled(self.w, self.h)
            self.labelLegCam.setPixmap(self.legPixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
    client_socket.close()