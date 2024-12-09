from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from Modules import *
import cv2
import sys
import socket
# import pyqtgraph as pg
from datetime import datetime

form_class = uic.loadUiType("CASS_ui.ui")[0]

# ESP32_IP = "172.20.10.8"
# ESP32_PORT = 8080
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

class MainWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setCamThreads()
        self.TCP = TCP()

        self.duration = 0
        self.maxDiff = 200
        self.authentified = False
        self.stopFlag = False
        self.isPersonAppear = False
        self.isRedLight = False
        self.isGreenLight = False

        # self.TCP.connect()

        self.ath_model = FaceRecognitionModel() # face athentification model
        self.setUserImage()

        self.detect_model = ObjectDetectionModel()
        self.segment_model = LaneSegmentation()
    
        self.setLabelCams()
        self.cameraOn()

        #for testing
        self.btnCamOn.clicked.connect(self.legCamOn)
        self.btnConnect.clicked.connect(self.connectLeg)
        self.btnTest.clicked.connect(self.test)

    # def setPlots(self):
    #     self.plotDiff = self.findChild(pg.PlotWidget, 'plotDiff')
    #     self.plotPerson = self.findChild(pg.PlotWidget, 'plotPerson')

    #     self.plotDiff.setBackground(background=None)
    #     self.plotPerson.setBackground('w')

        # self.plotPerson.showGrid(x=True, y=True)

    # def updatePlots(self):
    #     self.data_line = self.plotDiff.plot(self.pen='y')

    def test(self):
        print("test env")
        self.btnTest.hide()
        self.btnConnect.hide()
        self.legCamOn()
        self.btnCamOn.hide()
        self.authentified = True

    def connectLeg(self):
        self.TCP.connectLeg()
        self.TCP.sendMsg("connect")
        self.btnConnect.hide()

    def setCamThreads(self):
        self.faceCam = Camera()
        self.daemon = True
        self.faceCam.update.connect(self.updateFaceCam)

        self.legCam = Camera()
        self.daemon = True
        self.legCam.update.connect(self.updateLegCam2)
        # self.legCam.update.connect(self.updatePlots)
    
    def setLabelCams(self):
        self.width, self.height = self.labelFace.width(), self.labelFace.height()
        self.facePixmap = QPixmap(self.width, self.height)

        self.w, self.h = self.labelLegCam.width(), self.labelLegCam.height()
        self.legPixmap = QPixmap(self.w, self.h)

    def cameraOn(self):
        '''
        turn a camera on when driver authentification is success
        '''
        if not self.authentified:
            self.start = time.time()
            self.faceCam.start()
            self.faceCam.isRunning = True
            self.faceVideo = cv2.VideoCapture(2)
            print("camera on")

        elif self.authentified:
            self.labelFace.hide()
            print("authentification success")
            self.legCam.start()
            self.legCam.isRunning = True
            self.legVideo = cv2.VideoCapture(0)

    def legCamOn(self):
        '''
        in case not using face_recognition
        '''
        message = self.name
        response = self.TCP.sendMsg(message)

        self.labelFace.hide()
        self.legCam.start()
        self.legCam.isRunning = True
        self.legVideo = cv2.VideoCapture(0)

        self.authentified = True

    def setUserImage(self):
        '''
        TODO:needkk to create user image
        '''
        self.path = "../../../test/data/face/my_img/soyoung.png"
        self.name = "soyoung"

        self.ath_model.set_user_image(self.path)
        self.ath_model.set_known_user(self.ath_model.my_face_encoding, self.name)
        
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

            if self.authentified == False:
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

    def objectDetection(self, results, frame):
        class_names = []
        widths = []
        boxes = []  
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            confidence = result.conf[0]
            class_id = int(result.cls[0])
            label = f"{self.detect_model.names[class_id]}: {confidence:.2f}"
            obj = self.detect_model.names[class_id]
            label = f"{obj}: {confidence:.2f}"
            color_class = self.detect_model.color_finder(obj)

            cv2.rectangle(frame, (x1, y1), (x2, y2), self.detect_model.color_finder(self.detect_model.names[class_id]), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.detect_model.color_finder(self.detect_model.names[class_id]), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), 
                          color_class, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                        color_class, 2)

            ref_image_width = x2 - x1
            class_names.append(self.detect_model.names[class_id])
            class_names.append(obj)
            widths.append(ref_image_width)
            boxes.append((x1, y1, x2, y2))

        return class_names, widths, boxes

    def updateLegCam2(self):
        ret, self.frame = self.legVideo.read()

        if ret:
            # frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            frame = self.frame.copy()
            h, w, c = frame.shape
            diff_x = self.segment_model(frame)

            detect_list = []

            mtx = self.detect_model.mtx
            dist = self.detect_model.dist
            known_widths = self.detect_model.known_widths
            new_matrix, _ = cv2.getOptimalNewCameraMatrix(mtx, 
                                                          dist, 
                                                          (w, h), 1, (w, h))
            calibrated_frame = cv2.undistort(frame, 
                                             mtx, 
                                             dist, 
                                             new_matrix)
            frame_results = self.detect_model.model.predict(calibrated_frame, 
                                                            conf=0.55, verbose=False)
            focal_length_found = 520.925

            new_matrix, _ = cv2.getOptimalNewCameraMatrix(self.detect_model.mtx, self.detect_model.dist, (w, h), 1, (w, h))
            calibrated_frame = cv2.undistort(frame, self.detect_model.mtx, self.detect_model.dist, new_matrix)
            frame_results = self.detect_model.model.predict(calibrated_frame, conf=0.55, verbose=False)
            focal_length_found = 5.
            '''TODO: need to change focal_length_found data'''

            class_names, widths, boxes = self.detect_model.objectDetection(frame_results, calibrated_frame)
            class_names, widths, boxes = self.objectDetection(frame_results, calibrated_frame)

            for name, width, (x1, y1, x2, y2) in zip(class_names, widths, boxes):
                if name in self.detect_model.known_widths:
                    distance = self.detect_model.distance_finder(focal_length_found, self.detect_model.known_widths[name], width) - 16
                    cv2.putText(calibrated_frame, f"{round(distance, 2)} cm", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.detect_model.color_finder(name), 2)

                qImg = QImage(calibrated_frame, w, h, w*c, QImage.Format_RGB888)

                if name in known_widths:
                    distance = self.detect_model.distance_finder(focal_length_found, 
                                                                 known_widths[name], 
                                                                 width) - 16
                    cv2.putText(calibrated_frame, f"{round(distance, 2)} cm", (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                                self.detect_model.color_finder(name), 2)
                    
                    detect_list.extend([name, distance])
                    
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
                    
            cvt_color_frame = cv2.cvtColor(calibrated_frame, cv2.COLOR_BGR2RGB)
            qImg = QImage(cvt_color_frame, w, h, w*c, QImage.Format_RGB888)
            self.legPixmap = self.legPixmap.fromImage(qImg)
            self.legPixmap = self.legPixmap.scaled(self.w, self.h)
            self.labelLegCam.setPixmap(self.legPixmap)
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
    # client_socket.close()