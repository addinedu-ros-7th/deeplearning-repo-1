from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import uic

import os
import cv2
import sys
import time
import asyncio
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_Center import *
from Modules import *

ui_file = "CASS_ui.ui"
register_file = "Register.ui"
form_class = uic.loadUiType(ui_file)[0]
form_register_class = uic.loadUiType(register_file)[0]

cam_index = False

legCamID = 0 if cam_index else 2
faceCamID = 2 if cam_index else 0

class MainWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("CASS Driving")

        self.setCamThreads()
        self.setStates()
        self.setLabels()
        self.setIcons()
        self.setBtns()

        self.driver = Driver()
        self.db = DataBase()
        self.TCP = TCP()

        self.setAiModels()
        self.setDriverImage()
        self.setLabelCams()
        self.cameraOn()

        self.setAsync()
        asyncio.ensure_future(self.connect_async())

    # Async
    def connectionRetry(self):
        asyncio.ensure_future(self.connect_async())

    def setAsync(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_events)
        self.timer.start(100)

        self.reader = None
        self.writer = None

        self.esp32_ip = self.TCP.esp32_ip
        self.esp32_port = self.TCP.esp32_port
    
    def process_events(self):
        loop = asyncio.get_event_loop()
        loop.call_soon(loop.stop)
        loop.run_forever()

    async def connect_async(self):
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.esp32_ip, self.esp32_port)
            self.sendMsg("connect") 
            print("Connected to ESP32")

        except Exception as e:
            print(f"An error occurred when connecting: {e}")

    async def async_send_message(self, message):
        try:
            self.writer.write(message.encode())
            await self.writer.drain()

            data = await self.reader.read(100)
            print(f'Received: {data.decode()}')

            if data:
                self.labelReceive.setText(data.decode())
            
            if data == message:
                self.stopFlag = True
                print("data received")
            else:
                self.stopFlag = False
                print("data not received")

        except Exception as e:
            pass

    def sendMsg(self, message):
        message = self.TCP.encodeMsg(message)
        if self.writer:
            asyncio.ensure_future(self.async_send_message(message))
        else:
            print("No writer available")

    async def close_connection(self):
        self.writer.close()
        await self.writer.wait_closed()
        print("Connection closed")

    def setAiModels(self):
        self.authentification_model = FaceRecognitionModel()
        self.drowsy_model = DrowsyDetection()
        self.detect_model = ObjectDetection('bestDetect.pt')
        self.segment_model = LaneSegmentation('bestSeg.pt')
        self.emergency_model = EmergencyRecognizer().cuda()
        self.emergency_model.load_state_dict(torch.load('emergency.pt'))
        self.emergency_model.RTparameter_setting()

    # set drive info
    def setDirection(self, direction):
        self.direction = direction
        
    def selectRoad(self, road):
        self.road = road

    def driveState(self):
        """
        Power button for auto driving
        """
        if self.isDrive == False:
            self.isDrive = True
            self.btnPower.setText("OFF")
        else:
            self.isDrive = False
            self.btnPower.setText("ON")
            # self.sendMsg("drive")

    def keyPressEvent(self, event):
        """
        control with keyboard

        when auto driving mode (engine_state = ON)
        W : straight 
        A : left _ steering direction
        D : right - steering direction
        S : stop - engine_state = OFF

        when manual driving mode (engine_state = OFF)
        W : drive 
        A : left 
        D : right
        S : reverse

        O : change engine state 
        R : record
        C : connect to esp32 
        E : emergency stop
        """
        if event.key() == Qt.Key_W:
            if self.isDrive == False:
                self.sendMsg("drive")
                print("send drive")
            else:
                self.setDirection("straight")
                print("direction straight")

        elif event.key() == Qt.Key_A:
            if self.isDrive == False:
                self.sendMsg("L2")
            else:
                self.setDirection("left")

        elif event.key() == Qt.Key_D:
            if self.isDrive == False:
                self.sendMsg("R2")
            else:
                self.setDirection("right")

        elif event.key() == Qt.Key_S:
            if self.isDrive == False:
                self.isDrive = True
            else:
                self.sendMsg("reverse")

        elif event.key() == Qt.Key_O:
            self.driveState()
            
        elif event.key() == Qt.Key_R:
            self.clickRecord()

        elif event.key() == Qt.Key_C:
            self.connectLeg()

        elif event.key() == Qt.Key_V:
            self.legCamOn()

        elif event.key() == Qt.Key_E:
            self.selectRoad("side_park")

    # Setting GUI
    def setBtns(self):
        self.btnRec.clicked.connect(self.clickRecord)
        self.btnRegister.clicked.connect(self.register)

        self.btnPower.clicked.connect(self.driveState)
        self.btnForward.clicked.connect(lambda: self.setDirection("straight"))
        self.btnRight.clicked.connect(lambda: self.setDirection("right"))
        self.btnLeft.clicked.connect(lambda: self.setDirection("left"))
        self.btnStop.clicked.connect(lambda: self.setDirection("stop"))

        self.btnConnect.clicked.connect(self.connectionRetry)

    def setIcons(self):
        self.btnForward.setIcon(QIcon("./icons/arrowUp.png"))
        self.btnRight.setIcon(QIcon("./icons/arrowRight.png"))
        self.btnLeft.setIcon(QIcon("./icons/arrowLeft.png"))
        self.btnBackward.setIcon(QIcon("./icons/arrowDown.png"))

        self.labelTrafficLightR.setPixmap(QPixmap("./icons/trafficLightR.jpg"))
        self.labelTrafficLightG.setPixmap(QPixmap("./icons/trafficLightG.jpg"))
        self.labelTrafficLight.setPixmap(QPixmap("./icons/trafficLight.jpg"))

        self.labelDynamicObstacleON.setPixmap(QPixmap("./icons/dynamicObstacleOn.jpg"))
        self.labelDynamicObstacleOFF.setPixmap(QPixmap("./icons/dynamicObstacle.jpg"))
        self.labelStaticObstacleON.setPixmap(QPixmap("./icons/staticObstacleOn.jpg"))
        self.labelStaticObstacleOFF.setPixmap(QPixmap("./icons/staticObstacle.jpg"))

        self.labelDirection.setPixmap(QPixmap("./icons/direction.png"))
        self.labelDirectionBack.setPixmap(QPixmap("./icons/directionBack.png"))
        self.labelDirectionStraight.setPixmap(QPixmap("./icons/directionStraight.png"))
        self.labelDirectionLeft.setPixmap(QPixmap("./icons/directionLeft.png"))
        self.labelDirectionRight.setPixmap(QPixmap("./icons/directionRight.png"))

        self.labelSelectRoad.setPixmap(QPixmap("./icons/selectRoad.png"))
        self.labelSelectRoadLeft.setPixmap(QPixmap("./icons/selectRoadLeft.png"))
        self.labelSelectRoadRight.setPixmap(QPixmap("./icons/selectRoadRight.png"))

    def setLabels(self):
        self.labelRec.hide()

        self.labelTrafficLightR.hide()
        self.labelTrafficLightG.hide()

        self.labelDynamicObstacleON.hide()
        self.labelStaticObstacleON.hide()

        self.labelDirectionBack.hide()
        self.labelDirectionStraight.hide()
        self.labelDirectionLeft.hide()
        self.labelDirectionRight.hide()

        self.labelSelectRoadLeft.hide()
        self.labelSelectRoadRight.hide()

    def setStates(self):
        self.duration = 0
        self.count = 0
        self.authentified = False
        self.stopFlag = False
        self.isDrive = False
        self.isRecording = False

        self.isDrowsy1 = False
        self.isDrowsy2 = False

        self.UI_objs = {'person':'dynamic', 
                        'obstacle':'static', 
                        'goat':'dynamic',
                        'red_light':'red',
                        'green_light':'green'}
        self.objs_keys = self.UI_objs.keys()

        # Lane Segmentation Parameter
        self.st = time.time()
        self.et = time.time()
        self.dt = 0
        self.avoid_check = False
        self.road_select = 'center'
        self.direction = "straight"
        self.road = "center"
        self.order = None
        self.curFlag = None
    
    def updateUI(self, objs, direction, select_road):     
        objs = set(objs)
        objs = [self.UI_objs[obj] for obj in objs if obj in self.objs_keys]

        if 'dynamic' in objs:
            self.labelDynamicObstacleON.show()
        else:
            self.labelDynamicObstacleON.hide()
        if 'static' in objs:
            self.labelStaticObstacleON.show()
        else:
            self.labelStaticObstacleON.hide()

        if 'red' in objs:
            self.labelTrafficLightR.show()
        else:
            self.labelTrafficLightR.hide()
        if 'green' in objs:
            self.labelTrafficLightG.show()
        else:
            self.labelTrafficLightG.hide()

        if 'straight' == direction:
            self.labelDirectionStraight.show()
        else:
            self.labelDirectionStraight.hide()

        if 'left' == direction:
            self.labelDirectionLeft.show()
        else:
            self.labelDirectionLeft.hide()
        if 'right' == direction:
            self.labelDirectionRight.show()
        else:
            self.labelDirectionRight.hide()
        if 'reverse' == direction:
            self.labelDirectionBack.show()
        else:
            self.labelDirectionBack.hide()

        if 'left' == select_road:
            self.labelSelectRoadLeft.show()
        else:
            self.labelSelectRoadLeft.hide()
        if 'right' == select_road:
            self.labelSelectRoadRight.show()
        else:
            self.labelSelectRoadRight.hide()

    # Camera
    def setCamThreads(self):
        self.faceCam = Thread()
        self.daemon = True
        self.faceCam.update.connect(self.updateFaceCam)
        self.faceCam.update.connect(self.emergencyDetection)

        self.legCam = Thread()
        self.daemon = True
        self.legCam.update.connect(self.updateLegCam)

        self.record = Thread()
        self.record.daemon = True
        self.record.update.connect(self.updateRecording)
    
    def setLabelCams(self):
        self.labelRec.hide()

        self.width, self.height = self.labelFace.width(), self.labelFace.height()
        self.facePixmap = QPixmap(self.width, self.height)

        self.w, self.h = self.labelLegCam.width(), self.labelLegCam.height()
        self.legPixmap = QPixmap(self.w, self.h)

    def cameraOn(self):
        '''
        turn a camera on when driver authentification is success
        '''
        if not self.athentified:
            self.start = time.time()
            self.faceCam.start()
            self.faceCam.isRunning = True
            self.faceVideo = cv2.VideoCapture(faceCamID)
            print("face camera on")

        elif self.athentified:
            self.labelFace.hide()
            print("authentification success")
            self.legCam.start()
            self.legCam.isRunning = True
            self.legVideo = cv2.VideoCapture(legCamID)

    def legCamOn(self):
        '''
        in case not using face_recognition
        '''
        self.labelFace.hide()
        self.legCam.start()
        self.legCam.isRunning = True
        self.legVideo = cv2.VideoCapture(legCamID)
        print("leg camera on")

        self.authentified = True

    # Record
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
            frame = cv2.cvtColor(self.leg_frame, cv2.COLOR_BGR2RGB)
            self.writer.write(frame)

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

        video_path = "../../CASS_driving_log/driveVideo/"
        os.makedirs(video_path, exist_ok=True)
        self.now = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = video_path + self.now + '.avi'
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

        w = int(self.legVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.legVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.writer = cv2.VideoWriter(file_name, self.fourcc, 20.0, (w, h))

    def recordingStop(self):
        print("record released")
        self.record.running = False
        self.labelRec.hide()

        if self.isRecording == True:
            self.writer.release()
            self.labelRec.hide()

    # driver register
    def register(self):
        self.registerWindow = QDialog(self)
        uic.loadUi("Register.ui", self.registerWindow)
        self.registerWindow.btnUserRegister.clicked.connect(self.driverRegister)
        self.registerWindow.btnReturn.clicked.connect(self.returnMain)
        self.registerWindow.editContactNumber.textChanged.connect(self.legExCheck)
        self.registerWindow.show()
        self.db.connectLocal()

    def createDriverImage(self, driver_id, name):
        img_path = "../../CASS_driving_log/driverImg/"
        os.makedirs(img_path, exist_ok=True)
        self.path = str(driver_id) + name
        cv2.imwrite(img_path + self.path + ".png", cv2.cvtColor(self.face_frame, cv2.COLOR_BGR2RGB))
        self.db.registerPhoto(driver_id, self.path)
        self.db.local.close()

    def driverRegister(self):
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
            
            if self.duration == 2:
                self.createDriverImage(self.driver_id, name)
                self.start = time.time()

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

    def fetchDriverInfo(self):
        self.db.connectLocal()
        self.driver.id, self.driver.name, \
            self.driver.birth, self.driver.contact = self.db.fetchUser(self.path)
    
    # driver authentification
    def setDriverImage(self):
        '''
        TODO: for multi recognition
        '''
        img_path = "../../CASS_driving_log/driverImg/"
        img = os.listdir(img_path)
        self.path = img
        self.fetchDriverinfo()
        self.name = self.driver.name

        self.authentification_model.set_user_image(self.path)
        self.authentification_model.set_known_user(self.authentification_model.my_face_encoding, self.name)

    def authentification(self, frame):
        '''
        authentify a driver face duration 3 secs
        '''
        self.face_locations, self.face_names = self.authentification_model.face_athentication(frame)

        if self.face_names == []:
            self.face_names = ["unknown"]

        if self.face_names[0] == self.name:
            if self.authentified == False:
                self.authentification_model.draw_boxes(frame, 
                self.face_locations, self.face_names)
                
                if self.duration > 3 and self.duration < 4:
                    self.authentified = True
                    QMessageBox.warning(self, "Authentification ", 
                    f"{self.name} Driver Authentification Success.")
                    self.sendMsg(self.name)
                        
                    self.labelFace.hide()
                    self.cameraOn()

    def drowsyDetection(self, frame):
        """
        if driver is drowsy, turn a camera on 
        if driver is drowsy for (5 secs), control CASS_leg for wake up driver
        """
        try:
            predict = self.drowsy_model(frame)

            if predict == 0:
                if self.duration > 3:
                    self.labelLegCam.setStyleSheet("border: 5px solid red")
                    self.isDrowsy1 = True
                    if self.isRecording == False:
                        self.clickRecord()
                        """
                        TODO: log to DB
                        """
                    if self.duration > 5:
                        self.sendMsg("emergency")
            else:
                self.start = time.time()
                self.labelLegCam.setStyleSheet("")
                self.isDrowsy1 = False
                if self.isRecording == True:
                    self.clickRecord()
        except:
            self.start = time.time()
            self.isDrowsy1 = False
            self.labelFaceCam.setStyleSheet("border: 1px solid white")

    def emergencyDetection(self):
        road_state = self.emergency_model.RTstreaming()
        if road_state == "Emergency":
            self.selectRoad("side_park")
        else:
            pass

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

            # if self.authentified == False:
            #     self.authentification(frame)
            
            # else:
            #     self.drowsyDetection(frame)
            
            # if self.isDrowsy1 != self.isDrowsy2:
            #     self.sendMsg("emergency")

            """
            TODO: side_park
            """

            h, w, c = frame.shape
            qImg = QImage(frame, w, h, w*c, QImage.Format_RGB888)
            self.facePixmap = self.facePixmap.fromImage(qImg)
            self.facePixmap = self.facePixmap.scaled(self.width, self.height)
            self.labelFaceCam.setPixmap(self.facePixmap)

    def updateLegCam(self):
        """
        lane segmentation for select road and object detection for driving state
        """
        ret, self.frame = self.legVideo.read()
        
        if ret:
            frame = self.frame.copy()
            h, w, c = frame.shape
            try:
                self.object_detection, frame = self.detect_model(frame)
                obstacle = self.object_detection["obstacle"]
                order = self.object_detection["order"]
                self.cls_list = self.object_detection["cls_list"]
            except:
                pass
            try:
                '''
                    # direction : straight, left, right
                    # select_road : center, left, right, side_park
                '''
                self.road_segment = self.segment_model(frame, self.direction, self.road, obstacle)
                
                slope = self.road_segment["slope"]
                avoid = self.road_segment["avoid"]
                is_side = self.road_segment["is_side"]

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

            response = 'drive'

            if self.isDrive == True:
                if order == 'drive':
                    try:
                        if slope:
                            th = self.threshold
                            th_2 = self.threshold_2
                            th_r = th
                            th_l = - th

                            if slope > th_l and slope < th_r :
                                response = "drive"
                            elif slope > th_r:
                                response = "R1"
                                if slope > th_r + 20:
                                    response = "R3"

                            elif slope < th_l:
                                response = "L1"
                                if slope < th_l - -20:
                                    response = "L3"
                            print(slope, response)
                        else:
                            response = "drive"
                    except:
                        response = "no driveway"
                else:
                    response = "stop"
            else:
                response = "stop"

            message = self.TCP.encodeMsg(response)
            self.sendMsg(message)
            self.updateUI(self.cls_list, self.direction, self.road_select)

            self.leg_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qImg = QImage(self.leg_frame, w, h, w*c, QImage.Format_RGB888)
            self.legPixmap = self.legPixmap.fromImage(qImg)
            self.legPixmap = self.legPixmap.scaled(self.w, self.h)
            self.labelLegCam.setPixmap(self.legPixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.ensure_future(asyncio.sleep(0)))

    sys.exit(app.exec_())