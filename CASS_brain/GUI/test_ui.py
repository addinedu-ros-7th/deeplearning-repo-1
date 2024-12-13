from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5 import uic
from Modules import *

import os 
import time
import glob
import cv2
import sys
import socket
# import pyqtgraph as pg
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_Center import ObjectDetection, LaneSegmentation, EmergencyRecognizer


print(os.getcwd())

path="/CASS_brain/GUI/"
ui_file = os.path.join(path, "CASS_ui.ui")
ui_file = "CASS_ui.ui"
register_file = os.path.join(path, "Register.ui")
register_file = "Register.ui"

form_class = uic.loadUiType(ui_file)[0]
form_register_class = uic.loadUiType(register_file)[0]

# ESP32_IP = "172.20.10.8"
# ESP32_PORT = 8080
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

index = True

legCamID = 0 if index else 2
faceCamID = 2 if index else 0

class MainWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("CASS Driving")

        self.setCamThreads()
        self.setStates()
        self.setBtns()

        self.driver = Driver()
        self.db = DataBase()
        self.TCP = TCP()

        self.ath_model = FaceRecognitionModel() # face athentification model
        self.setUserImage()

        self.detect_model = ObjectDetection('bestDetect.pt')
        self.segment_model = LaneSegmentation('bestSeg.pt')
        self.emergency_model = EmergencyRecognizer().cuda()
        self.emergency_model.load_state_dict(torch.load('bestEmergency.pt'))
    
        self.setLabelCams()
        self.cameraOn()

    # def setPlots(self):
    #     self.plotDiff = self.findChild(pg.PlotWidget, 'plotDiff')
    #     self.plotPerson = self.findChild(pg.PlotWidget, 'plotPerson')

    #     self.plotDiff.setBackground(background=None)
    #     self.plotPerson.setBackground('w')

        # self.plotPerson.showGrid(x=True, y=True)

    # def updatePlots(self):
    #     self.data_line = self.plotDiff.plot(self.pen='y')

    def setStates(self):
        self.duration = 0
        self.maxDiff = 200
        self.count = 0
        self.authentified = False
        self.stopFlag = False
        self.isDrive = False
        self.isPersonAppear = False
        self.isRedLight = False
        self.isGreenLight = False
        self.isRecording = False

        # Lane Segmentation Parameter
        self.st = time.time()
        self.et = time.time()
        self.dt = 0
        self.avoid_check = False
        self.checkitout = True
        self.road_select = 'center'
        self.direction = "straight"
        self.road = "center"
        self.order = None
        self.prev_order = None
        self.curFlag = None

    def setBtns(self):
        self.labelRec.hide()

        self.btnRec.clicked.connect(self.clickRecord)
        self.btnRegister.clicked.connect(self.register)

        self.btnForward.clicked.connect(lambda: self.setDirection("straight"))
        self.btnRight.clicked.connect(lambda: self.setDirection("right"))
        self.btnLeft.clicked.connect(lambda: self.setDirection("left"))
        self.btnStop.clicked.connect(lambda: self.setDirection("stop"))

        # if self.isDrive == False:
        self.btnPower.clicked.connect(self.driveState)
        # else:
            # self.btnRight.clicked.connect(lambda: self.road("right")) # 추월용 비키삼~!
            # self.btnLeft.clicked.connect(lambda: self.road("left"))
            # self.btnBackward.clicked.connect(lambda: self.road("back"))

        #for testing
        self.btnCamOn.clicked.connect(self.legCamOn)
        self.btnConnect.clicked.connect(self.connectLeg)
        self.btnTest.clicked.connect(self.test)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_W:
            self.setDirection("straight")
            print(event)
        elif event.key() == Qt.Key_A:
            self.setDirection("left")
        elif event.key() == Qt.Key_D:
            self.setDirection("right")
        elif event.key() == Qt.Key_S:
            self.setDirection("stop")
        elif event.key() == Qt.Key_O:
            self.driveState()

    def setDirection(self, direction):
        self.direction = direction
        self.labelDirection.setText(direction)
        print("set direction to the ", direction)

    def driveState(self):
        if self.isDrive == False:
            self.isDrive = True
        else:
            self.isDrive = False

        print("leg has started driving ? ", self.isDrive)

    def test(self):
        print("test env")
        self.btnTest.hide()
        self.btnConnect.hide()
        self.legCamOn()
        self.btnCamOn.hide()
        self.authentified = True

    def clickRecord(self):
        if self.isRecording == False:
            self.labelRec.show()
            self.isRecording = True
            self.recordingStart()
        else:
            self.labelRec.hide()
            self.isRecording = False
            self.recordingStop()

    def updateRecording(self):
        if self.isRecording == True:
            self.count += 1
            self.writer.write(self.leg_frame)

            if self.count % 8 == 0:
                self.labelRec.hide()
            else:
                self.labelRec.show()
        else:
            self.labelRec.hide()
            self.count = 0

    def recordingStart(self):
        print("record started")
        self.record.running = True
        self.record.start()

        self.now = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = '../../../test/data/face/drive/' + self.now + '.avi'
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

        w = int(self.legVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.legVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.writer = cv2.VideoWriter(filename, self.fourcc, 20.0, (w, h))

    def recordingStop(self):
        print("record released")
        self.record.running = False
        self.labelRec.hide()

        if self.isRecording == True:
            self.writer.release()
            self.labelRec.hide()

    def connectLeg(self):
        self.TCP.connectLeg()
        self.TCP.sendMsg("connect")
        self.btnConnect.hide()

    def setCamThreads(self):
        self.faceCam = Camera()
        self.faceCam.daemon = True
        self.faceCam.update.connect(self.updateFaceCam)

        self.legCam = Camera()
        self.legCam.daemon = True
        self.legCam.update.connect(self.updateLegCam2)
        # self.legCam.update.connect(self.updatePlots)

        self.record = Camera()
        self.record.daemon = True
        self.record.update.connect(self.updateRecording)
    
    def setLabelCams(self):
        self.width, self.height = self.labelFaceCam.width(), self.labelFaceCam.height()
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
            self.faceVideo = cv2.VideoCapture(faceCamID)
            print("camera on")

        elif self.authentified:
            self.labelFaceCam.hide()
            print("authentification success")
            self.legCam.start()
            self.legCam.isRunning = True
            self.legVideo = cv2.VideoCapture(legCamID)

    def legCamOn(self):
        '''
        in case not using face_recognition
        '''
        message = self.name
        response = self.TCP.sendMsg(message)

        self.labelFaceCam.hide()
        self.legCam.start()
        self.legCam.isRunning = True
        self.legVideo = cv2.VideoCapture(legCamID)

        self.authentified = True

    def setUserImage(self):
        path_dir = "../../../test/data/face/register"
        file_name = os.listdir(path_dir)
        path_name = file_name[0][:-4]
        '''
        TODO:need to create user image
        '''
        self.path = "../../../test/data/face/test_img/img_sb.png"
        self.name = "soyoung"

        self.ath_model.set_user_image(self.path)
        self.ath_model.set_known_user(self.ath_model.my_face_encoding, self.name)

    def register(self):
        self.registerWindow = QDialog(self)
        uic.loadUi("Register.ui", self.registerWindow)
        self.registerWindow.btnUserRegister.clicked.connect(self.userRegister)
        self.registerWindow.btnReturn.clicked.connect(self.returnMain)
        self.registerWindow.editContactNumber.textChanged.connect(self.legExCheck)
        self.registerWindow.show()
        self.db.connectLocal()

    def userRegister(self):
        name = self.registerWindow.editName.text()
        birth = self.registerWindow.dateEditBirth.date().toPyDate()
        contact_number = self.registerWindow.editContactNumber.text()

        if name == "" or birth == "" or contact_number == "":
            QMessageBox.warning(self, "warning", "빈칸을 채워주세요")
            return
        
        else:
            self.driver.getInfo(name, birth, contact_number)
            self.registerWindow.hide()
            self.driver_id = self.db.registerUser(name, birth, contact_number)
            QMessageBox.warning(self, "Photo", "정면을 보세요~ 2초 뒤 사진이 촬영됩니다.")

            time.sleep(2)
            self.createUserImage(self.driver_id, name)

    def createUserImage(self, driver_id, name):
        dir = "../../../test/data/face/register/"
        self.path = str(driver_id) + name
        cv2.imwrite(dir + self.path + ".png", cv2.cvtColor(self.face_frame, cv2.COLOR_BGR2RGB))
        self.db.registerPhoto(driver_id, self.path)
        self.db.local.close()

    def legExCheck(self):
        tmp = self.registerWindow.editContactNumber.text()
        if len(tmp) == 3 and self.len_prev == 2:
            self.registerWindow.editContactNumber.setText(tmp + "-")
        elif len(tmp) == 8 and self.len_prev == 7:
            self.registerWindow.editContactNumber.setText(tmp + "-")
        elif len(tmp) == 8 and self.len_prev == 9:
            self.registerWindow.editContactNumber.setText(tmp + "-")
        self.len_prev = len(tmp)

    def returnMain(self):
        self.registerWindow.hide()
        
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
                        
                    self.labelFaceCam.hide()
                    self.cameraOn()

    def updateFaceCam(self):
        """
        for face authentification and if authentified for drowsy detection
        """
        ret, self.face_frame = self.faceVideo.read()
        
        if ret:
            frame = cv2.cvtColor(self.face_frame, cv2.COLOR_BGR2RGB)
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
            self.labelFaceCam.setPixmap(self.facePixmap)

    def updateLegCam(self):
        ret, self.frame = self.legVideo.read()
        
        if ret:
            frame = self.frame.copy()
            h, w, c = frame.shape
            diff_x = self.segment_model(frame)
            detect_list = self.detect_model.get_distance(frame)

            
            # if len(detect_list) != 0:
            #     print(detect_list)

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
                # print("diff_x : ", diff_x)

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

    def updateLegCam2(self):
        ret, self.frame = self.legVideo.read()
        
        if ret:
            # frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            frame = self.frame.copy()
            h, w, c = frame.shape
            try:
                order, frame, obstacle = self.detect_model(frame)
            except:
                pass
            try:
                '''
                    # direction : straight, left, right
                    # select_road : center, left, right
                '''
                slope, avoid, is_side = self.segment_model(frame, self.direction, self.road, obstacle)

                if not(is_side) and self.road != 'center':
                    self.road = 'center'
                if avoid != self.avoid_check and self.check:
                    self.check = False
                    self.st = time.time()
                    if avoid:
                        self.avoid_check = avoid
                        self.road_select = 'left'
                    else:
                        self.road_select = 'center'
                        self.avoid_check = avoid

                if self.avoid_check:
                    self.et = time.time()
                    self.dt = self.et - self.st
                    if self.dt > 2 and self.dt <= 6:
                        self.road_select = 'right'
                    elif self.dt > 6:
                        self.st = time.time()
                        self.check = True
            except:
                pass
                
            if self.isDrive == True:
                if order == 'straight' or order == 'Avoidance':
                    try:
                        if slope:
                            th = 30
                            th_r = th/2
                            th_l = - (th/2)

                            if slope < th_l or slope > th_r :
                                response = "drive"
                            elif slope > th_r:
                                response = "R1"
                                if slope > th_r + th:
                                    response = "R2"
                            elif slope < th_l:
                                response = "L1"
                                if slope < th_l - th:
                                    response = "L2"
                        else:
                            response = "drive"
                    except:
                        # print("no driveway")
                        response = "no driveway"
                else:
                    response = "stop"
            else:
                response = "stop"

            if response != self.curFlag and response != "no driveway":
                print("response : ", response)
                self.curFlag = response
                self.TCP.sendMsg(response)
            
            self.labelState.setText(response)
            self.labelDirection.setText(self.direction)

            self.leg_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qImg = QImage(self.leg_frame, w, h, w*c, QImage.Format_RGB888)
            self.legPixmap = self.legPixmap.fromImage(qImg)
            self.legPixmap = self.legPixmap.scaled(self.w, self.h)
            self.labelLegCam.setPixmap(self.legPixmap)

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
    # client_socket.close()