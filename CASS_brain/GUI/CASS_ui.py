from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
from Modules import *
import cv2
import sys
import socket
from datetime import datetime

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
        self.authentified = False
        self.isPersonAppear = False
        self.isRedLight = False
        self.isGreenLight = False

        self.setAiModels()
        self.setUserImage()
        self.setLabelCams()
        self.cameraOn()

        #for testing
        self.btnSend.clicked.connect(self.legCamOn)
        self.lineEdit.returnPressed.connect(self.sendData)
    
    def setAiModels(self):
        self.ath_model = FaceRecognitionModel()
        self.detect_model = ObjectDetectionModel()
        self.segment_model = LaneSegmentation()

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

    def sendData(self):
        message = self.lineEdit.text()
        '''
        TODO: need to solve error triggerd by "\msg\n"
        '''
        msg = message.encode()
        client_socket.send(msg)
        print("CASS_brain said : ", msg)
        response = client_socket.recv(1024)
        print("CASS_leg said : ", response)

    def sendMsg(self, msg):
        client_socket.send(msg)
        print("CASS_brain said : ", msg)
        response = client_socket.recv(1024)
        print("CASS_leg said : ", response)

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
            self.faceVideo = cv2.VideoCapture(2)
            print("camera on")

        elif self.athentified:
            self.labelFace.hide()
            print("authentification success")
            self.legCam.start()
            self.legCam.isRunning = True
            self.legVideo = cv2.VideoCapture(0)

    def legCamOn(self):
        '''
        in case not using face_recognition
        '''
        self.labelFace.hide()
        self.legCam.start()
        self.legCam.isRunning = True
        self.legVideo = cv2.VideoCapture(0)

        self.authentified = True

    def authentification(self, frame):
        '''
        authentify a driver face duration 3 secs
        '''
        self.face_locations, self.face_names = self.ath_model.face_athentication(frame)

        if self.face_names == []:
            self.face_names = ["unknown"]

        if self.face_names[0] == self.name:
            if self.authentified == False:
                self.ath_model.draw_boxes(frame, 
                self.face_locations, self.face_names)
                
                if self.duration > 3 and self.duration < 4:
                    self.authentified = True
                    QMessageBox.warning(self, "Authentification ", 
                    f"{self.name} Driver Authentification Success.")
                    response = self.TCP.sendMsg(str(self.name))
                    print(self.name)
                    print("auth response : ", response)
                        
                    self.labelFace.hide()
                    self.cameraOn()
                    
    def updateFaceCam(self):
        ret, face_frame = self.faceVideo.read()
        
        if ret:
            frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            
            self.end = time.time()
            self.duration = self.end - self.start

            self.authentification(frame)
            
            # else:
            # self.DrowsyDetection(frame)
            
            # if self.isDrowsy1 != self.isDrowsy2:
                # self.send_Drowsy()
                
            h, w, c = frame.shape
            qImg = QImage(frame, w, h, w*c, QImage.Format_RGB888)
            self.facePixmap = self.facePixmap.fromImage(qImg)
            self.facePixmap = self.facePixmap.scaled(self.width, self.height)
            self.labelFace.setPixmap(self.facePixmap)

    def updateLegCam(self):
        ret, self.frame = self.legVideo.read()
        
        if ret:
            frame = self.frame.copy()
            h, w, c = frame.shape
            diff_x = self.segment_model(frame)
            detect_list = self.detect_model.get_distance(frame)

            if len(detect_list) != 0:
                print(detect_list)

            if self.isPersonAppear == False:
                if "person"  in detect_list:
                    response = self.TCP.sendMsg("stop")
                    self.isPersonAppear = True

            else:
                if "person" not in detect_list:
                    response = self.TCP.sendMsg("drive")
                    self.isPersonAppear = False

            if self.isRedLight == False:
                if "red_light" in detect_list:
                    response = self.TCP.sendMsg("stop")
                    self.isRedLight = True
                    self.isGreenLight = False

            else:
                if "red_light" not in detect_list:
                    self.isRedLight = False
                    self.isGreenLight = True

            if self.isGreenLight == False:
                if "green_light" in detect_list:
                    response = self.TCP.sendMsg("drive")
                    self.isGreenLight = True
                    self.isRedLight = False

            if diff_x:
                print("diff_x : ", diff_x)

                if self.isPersonAppear == False:
                    if diff_x == 0:
                        response = self.TCP.sendMsg("drive")
                    elif diff_x > 50:
                        response = self.TCP.sendMsg("R1")
                        if diff_x > 180:
                            response = self.TCP.sendMsg("R2")
                    elif diff_x < 50:
                        response = self.TCP.sendMsg("L1")
                        if diff_x < -180:
                            response = self.TCP.sendMsg("L2")

                # data_line = self.plotDiff.plot(diff_x, pen='k', width=2)
                # print(data_line)
                # data_line.setData(diff_x)
                        
            cvt_color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qImg = QImage(cvt_color_frame, w, h, w*c, QImage.Format_RGB888)
            self.legPixmap = self.legPixmap.fromImage(qImg)
            self.legPixmap = self.legPixmap.scaled(self.w, self.h)
            self.labelLegCam.setPixmap(self.legPixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
    client_socket.close()