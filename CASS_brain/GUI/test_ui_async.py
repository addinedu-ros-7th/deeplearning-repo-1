from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5 import uic
from Modules import *

# 서버 설정
import os 
import cv2
import sys
import time
import asyncio
import threading
from datetime import datetime

lock = threading.Lock()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_Center import *

ui_file = "CASS_gui.ui"
register_file = "Register.ui"

form_class = uic.loadUiType(ui_file)[0]
form_register_class = uic.loadUiType(register_file)[0]

index = True
legCamID = 0 if index else 2
faceCamID = 2 if index else 0

class MainWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("CASS Driving")

        self.setThreads()
        self.setStates()
        self.setLabels()
        self.setIcons()
        self.setBtns()
        self.setLabelCams()
        self.cameraOn()

        self.driver = Driver()
        self.db = DataBase()
        self.db.connectLocal("****")
        self.TCP = TCP()

        self.ath_model = FaceRecognitionModel() # face athentification model
        self.setDriverImage()

        self.detect_model = ObjectDetection('bestDetect.pt')
        self.segment_model = LaneSegmentation('bestSeg.pt')
        self.voice_id_model = VoiceRecognizer(input_size=120, n_classes=6).cuda()
        self.voice_id_model.load_state_dict(torch.load('Voice_Check.pt'))

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_events)
        self.timer.start(100)

        self.reader = None
        self.writer = None

        asyncio.ensure_future(self.connect_async())

    def process_events(self):
        # loop = asyncio.get_event_loop()
        # loop.call_soon(loop.stop)
        # loop.run_forever()

        try:
            loop = asyncio.get_event_loop()
            loop.call_soon(loop.stop)
            loop.run_forever()
        except RuntimeError:
            # 현재 스레드에 이벤트 루프가 없는 경우 새로 생성
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            print("runtime error")

    async def connect_async(self):
        print("connecting...")
        try:
            self.reader, self.writer = await asyncio.open_connection(self.TCP.esp32_ip, self.TCP.esp32_port)
            print("Connected to ESP32")
            self.sendMsg("11\n")
        except asyncio.TimeoutError:
            print(f"Connection timed out!")
        except Exception as e:
            print(f"An error occurred when connecting: {e}")

    async def async_send_message(self, message):
        try:
            # async with lock:
            self.writer.write(message.encode())
            await self.writer.drain()

            data = await self.reader.read(100)

            if data:
                data_split = data.decode('utf-8').split('\n')
                print(f'Received from ESP32: {data_split}')

                if data_split[1] == "0":
                    self.isDrive = False
                    self.isPowerOn = False
                    self.labelState.setText("Power Off")
                    self.labelState2.setText("Stop")

        except Exception as e:
            # print("error {e} occured when sending messages")
            pass

    def sendMsg(self, message):
        if self.writer:
            asyncio.ensure_future(self.async_send_message(message))
        else:
            # print("No writer available push key_C for reconnect")
            pass

    async def close_connection(self):
        self.writer.close()
        await self.writer.wait_closed()
        print("Connection closed")

    def setThreads(self):
        self.faceCam = Thread() #
        self.faceCam.daemon = True #
        self.faceCam.update.connect(self.updateFaceCam) 

        self.legCam = Thread()
        self.legCam.daemon = True
        self.legCam.update.connect(self.updateLegCam)

        self.record = Thread()
        self.record.daemon = True
        self.record.update.connect(self.updateRecording)

        self.serverSocket = ServerSocket()
        self.serverSocket.daemon = True
        self.serverSocket.update.connect(self.get_socket_data)

    def setLabelCams(self):
        self.width, self.height = self.labelFaceCam.width(), self.labelFaceCam.height()
        self.facePixmap = QPixmap(self.width, self.height)

        self.w, self.h = self.labelLegCam.width(), self.labelLegCam.height()
        self.legPixmap = QPixmap(640, 480)

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
        self.labelSelectRoadSidepark.setPixmap(QPixmap("./icons/selectRoadSidepark.jpg"))

    def setLabels(self):
        self.labelRec.hide()
        self.labelThread.hide()

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
        self.labelSelectRoadSidepark.hide()

    def setStates(self):
        self.duration = 0
        self.count = 0
        self.authentified = False

        self.power_state = "off"
        self.drive_state = "stop"
        self.isPowerOn = False
        self.isDrive = False
        self.isReverse = False
        self.isRecording = False

        self.emergency = None
        self.isEmergency1 = False
        self.isEmergency2 = False

        self.isDrowsy1 = False
        self.isDrowsy2 = False

        self.UI_objs = {'person':'dynamic', 
                        'obstacle':'static', 
                        'goat':'dynamic',
                        'red_light':'red',
                        'green_light':'green'}
        self.objs_keys = self.UI_objs.keys()

        # Lane Segmentation Parameter
        self.threshold = 40
        self.count_nd = 0
        self.st = time.time()
        self.et = time.time()
        self.dt = 0
        self.avoid_check = False
        self.select_road = 'center'
        self.direction = "straight"
        self.order = None
        self.ab_order = None
        self.curFlag = None

    def setBtns(self):
        self.btnRegister.hide()

        self.btnPower.clicked.connect(self.driveState)
        self.btnForward.clicked.connect(lambda: self.setDirection("straight"))
        self.btnRight.clicked.connect(lambda: self.setDirection("right"))
        self.btnLeft.clicked.connect(lambda: self.setDirection("left"))
        self.btnStop.clicked.connect(lambda: self.setDirection("stop"))

        # self.btnRight.clicked.connect(lambda: self.road("right")) # 추월용 비키삼~!
        # self.btnLeft.clicked.connect(lambda: self.road("left"))
        # self.btnBackward.clicked.connect(lambda: self.road("back"))

    def keyPressEvent(self, event):
        """
        control with keyboard

        when auto driving mode
        W : straight 
        A : left _ steering direction
        D : right - steering direction
        S : stop

        when manual driving mode
        W : drive 
        A : left (L3)
        D : right (R3)
        S : reverse

        P : car power ON/OFF
        O : change drive state
        R : record
        C : connect to esp32
        V : voice authentification

        """
        if event.key() == Qt.Key_Q:
            if self.isAutoDrive == True:
                self.isAutoDrive = False
            else:
                self.isAutoDrive = True 

        elif event.key() == Qt.Key_W:
            if self.isAutoDrive == True:
                self.setDirection("straight")
            else:
                message = self.TCP.encodeMsg("drive")
                self.sendMsg(message)

        elif event.key() == Qt.Key_A:
            if self.isAutoDrive == True:
                self.setDirection("left")
            else:
                message = self.TCP.encodeMsg("L3")
                self.sendMsg(message)

        elif event.key() == Qt.Key_D:
            if self.isAutoDrive == True:
                self.setDirection("right")
            else:
                message = self.TCP.encodeMsg("R3")
                self.sendMsg(message)

        elif event.key() == Qt.Key_S:
            if self.isAutoDrive == True and self.isDrive == True:
                self.driveState()
            else:
                message = self.TCP.encodeMsg("reverse")
                self.sendMsg(message)

        elif event.key() == Qt.Key_R:
            self.clickRecord()

        # for testing
        elif event.key() == Qt.Key_T:
            self.test()

        elif event.key() == Qt.Key_C:
            self.connectLeg()

        elif event.key() == Qt.Key_O:
            self.driveState()
            
        elif event.key() == Qt.Key_P:
            if self.isPowerOn == True:
                self.isPowerOn = False
                self.sendMsg("23\n")
                response = 'off'
                self.labelState.setText("Power Off")
            else:    
                self.isPowerOn = True
                self.sendMsg("22\n")
                response = "on"
                self.labelState.setText("Power On")

            try:
                self.serverSocket.client_socket.send(response.encode('utf-8'))
                print(f"{response} has sent")
            except Exception as e:
                print(f"connection lost : {e}")

        elif event.key() == Qt.Key_V:
            self.voiceAuthentification()

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
        if 'side_park' == select_road:
            self.labelSelectRoadSidepark.show()
        else:
            self.labelSelectRoadSidepark.hide()

    def voiceAuthentification(self):
        if self.authentified == False:
            QMessageBox.warning(self, "Voice Authentification", "주행자 인증을 위한 3초 음성인식을 시작합니다.")
            time.sleep(0.4)
            result = self.voice_id_model.voicecheck()
            if result == True:
                self.labelThread.hide()
                self.authentified = True
                QMessageBox.information(self, "Voice Authentification", "주행자 인증이 완료되었습니다.")
                message = self.TCP.encodeMsg(self.name)
                self.sendMsg(message)
                self.cameraOn()
            else:
                QMessageBox.warning(self, "Voice Authentification", "인증에 실패하였습니다. 다시 인증해주세요.")
        else:
            self.labelThread.hide()
            pass
    
    def updateVoiceIdModel(self):
        return self.voice_id_model.voicecheck()

    def voice_order(self, value):
        if value == "on" and self.state1 == "off":
            self.power_state = "on"
            self.isPowerOn = True
            msg = self.TCP.encodeMsg("power_on")
            self.sendMsg(msg)
            self.labelState.setText("Power On")

        elif value == "off" and self.state1 == "on":
            self.power_state = "off"
            self.isPowerOn = False
            msg = self.TCP.encodeMsg("power_off")
            self.sendMsg(msg)
            self.labelState.setText("Power Off")

        elif value == "go" and self.state2 == "stop":
            self.drive_state = "go"
            self.driveState()
            self.setDirection("straight")
            self.labelState2.setText("Go")

        elif value == "stop" and self.state2 == "go":
            self.drive_state = "stop"
            self.driveState()
            self.labelState2.setText("Stop")

    def setDirection(self, direction):
        self.direction = direction

    def selectRoad(self, road):
        self.select_road = road

    def driveState(self):
        if self.isDrive == False:
            self.isDrive = True
            message = self.TCP.encodeMsg("accel")
            self.sendMsg(message)
            self.sendMsg(message)
            self.sendMsg(message)
            self.sendMsg(message)
            print("drive go, accel on")
            response = "go"
            self.labelState2.setText("Go")
        else:
            self.isDrive = False
            response = "stop"
            self.labelState2.setText("Stop")
            print("drive stop")

        self.serverSocket.client_socket.send(response.encode('utf-8'))

    def test(self):
        self.btnTest.hide()
        self.btnRegister.hide()
        self.legCamOn()
        self.sendMsg("21\n")
        self.authentified = True
        self.serverSocket.start()
        self.serverSocket.isRunning = True

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

    def cameraOn(self):
        '''
        turn a camera on when driver authentification is success
        '''
        if not self.authentified:
            self.start = time.time()
            self.faceCam.start() #
            self.faceCam.isRunning = True #
            self.faceVideo = cv2.VideoCapture(faceCamID)
            print("face camera on")

        elif self.authentified:
            self.labelFaceCam.hide()
            print("authentification success and leg camera on")
            self.legCam.start()
            self.legCam.isRunning = True
            self.legVideo = cv2.VideoCapture(legCamID)
            self.serverSocket.start()
            self.serverSocket.isRunning = True

    def legCamOn(self):
        '''
        in case not using face_recognition
        '''
        print('Leg Cam ON')
        self.sendMsg("21\n")

        self.labelFaceCam.hide()
        self.legCam.start()
        self.legCam.isRunning = True
        self.legVideo = cv2.VideoCapture(legCamID)

        self.authentified = True

    def connectLeg(self):
        """
        retry to connect with ESP32 by pressing Key_C
        """
        asyncio.ensure_future(self.connect_async())

    def setDriverImage(self):
        # self.db.fetchUserData(self.path)

        self.name = "soyoung"
        img_path = "../../CASS_driving_log/driverImg/"
        self.db.cur = self.db.local.cursor(buffered=True)
        sql = "SELECT * FROM DRIVER WHERE name = %s"
        self.db.cur.execute(sql, ([self.name]))
        data = self.db.cur.fetchall()
        self.driver_id = data[0][0]
        self.path = data[0][4]

        self.ath_model.set_user_image(img_path + self.path)
        self.ath_model.set_known_user(self.ath_model.my_face_encoding, self.name)

    def authentification(self, frame):
        '''
        authentify a driver face duration 3 secs
        '''
        self.face_locations, self.face_names = self.ath_model.face_athentication(frame)

        if self.face_names == []:
            self.face_names = ["unknown"]

        if self.name in self.face_names:
            if self.authentified == False:
                self.ath_model.draw_boxes(frame, 
                self.face_locations, self.face_names)
                
                if self.duration > 3 and self.duration < 4:
                    self.authentified = True
                    QMessageBox.information(self, "Authentification ", 
                    f"{self.name} Driver Authentification Success.")
                    message = self.TCP.encodeMsg(self.name)
                    self.sendMsg(message)
                    self.cameraOn()
                    print("authentification success")

    def get_socket_data(self, data):
        data = data.split(' : ')
        if len(data) == 1:
            # print(len(data))
            return
        else:
            key = data[0]
            value = data[1]
            response = "nak"

            if 'Bot' in key:
                print(f"Received from Bot: {value}")
                response = f"Server resend to Bot: {value}"
                self.voice_order(value)

            elif 'Drowsy' in key:
                print(f"Received from Drowsy: {value}")
                response = f"Server resend to Drowsy: {value}"
                self.drowsyDetected(value)

            elif 'Siren' in key:    
                print(f"Received from Siren: {value}")
                response = f"Server resend to Siren: {value}"
                self.updateEmergency(value)
                self.emergencyLogger()

            self.serverSocket.client_socket.send(response.encode('utf-8'))

    def drowsyDetected(self, state):
        """
        if driver is drowsy, turn a camera on 
        if driver is drowsy for (5 secs), control CASS_leg for wake up driver
        """
        if state == "Drowsy":
            self.isDrowsy1 = True
        else:
            self.isDrowsy1 = False
            self.isDrowsy2 = False
            self.start = time.time()
            self.labelLegCam.setStyleSheet("")
            if self.isRecording == True:
                self.clickRecord()

        if self.isDrowsy1 == True and self.isDrowsy2 == False:
            self.isDrowsy2 = True
            msg = self.TCP.encodeMsg("wake_up")
            self.sendMsg(msg)
            self.labelLegCam.setStyleSheet("border: 5px solid red")
            self.start = time.time()
        elif self.isDrowsy1 == True and self.isDrowsy2 == True:
            self.end = time.time()
            self.duration = self.end - self.start

            if self.duration > 3:
                msg = self.TCP.encodeMsg("side_parking")
                self.sendMsg(msg)
                self.selectRoad("side_park")

    def updateEmergency(self, value):
        if value == "Emergency":
            self.isEmergency1 = True
            self.select_road = "side_park"
            self.labelThread.show() 
            self.labelThread.setText("Emergency Situation")
            self.labelThread.setStyleSheet("background-color: red; color: white; font-size: 20px;")
        else:
            self.isEmergency1 = False
            self.isEmergency2 = False
            self.labelThread.hide()
            if self.select_road == "side_park":
                self.selectRoad("center")

    def emergencyLogger(self):
        if self.isEmergency1 != self.isEmergency2:
            sql = "INSERT INTO emergency_log (driver_id) VALUES (%s)"
            self.db.cur.execute(sql, [self.driver_id])
            print("hello")
            self.db.local.commit()
            print("emergency log committed")
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

            if self.authentified == False:
                self.authentification(frame)

            h, w, c = frame.shape
            qImg = QImage(frame, w, h, w*c, QImage.Format_RGB888)
            self.facePixmap = self.facePixmap.fromImage(qImg)
            self.facePixmap = self.facePixmap.scaled(self.width, self.height)
            self.labelFaceCam.setPixmap(self.facePixmap)

    def updateLegCam(self):
        """
        get object detection result and road segmentation for driving
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
                self.road_segment = self.segment_model(frame, self.direction, self.select_road, obstacle)
                
                slope = self.road_segment["slope"]
                avoid = self.road_segment["avoid"]
                is_side = self.road_segment["is_side"]

                if not(is_side) and self.select_road != 'center' and self.isEmergency1 == False:
                    self.select_road = 'center'
                
                if obstacle != None:
                    self.avoid_check = True
                    self.select_road = 'left'
                    self.st = time.time()
                else: 
                    if self.dt > 8 and self.dt < 10:
                        self.select_road = 'right'
                    elif self.dt > 1:
                        self.select_road = 'center'
                    # print(self.dt)

                    if self.dt > 14:
                        self.avoid_check = False
                        self.st = 0
                        self.et = 0
                        self.dt = 0
                \
                if self.avoid_check and obstacle==None:
                    self.et = time.time()
                    self.dt = self.et - self.st

            except:
                pass

            response = 'drive'

            if self.isDrive == True:
                if order == 'drive':
                    if "red_light" in self.cls_list:
                        self.ab_order = "stop"
                    elif "green_light" in self.cls_list:
                        self.ab_order = "drive"

                    try:

                        if slope:
                            th = self.threshold
                            th_r = th
                            th_l = - th

                            if slope > th_l and slope < th_r :
                                response = "drive"
                            elif slope > th_r:
                                response = "R1"
                                if slope > th_r + 20: 
                                    response = "R3"
                                    if self.select_road == "side_park" or self.select_road == "right":
                                        response = "R2"

                            elif slope < th_l:
                                response = "L1"
                                if slope < th_l - -20:
                                    response = "L3"
                                    if self.select_road == "left" or self.select_road == "side_park":
                                        response = "L2"
                                        
                            self.count_nd = 0
                        else:
                            response = "drive"
                            self.count_nd = 0
                    except:
                        response = "no_drive_way"
                        self.count_nd += 1
                        if self.count_nd > 20:
                            response = "reverse"
                else:
                    response = "stop"

            else:
                if self.isReverse == True:
                    response = "reverse"
                else:
                    response = "stop"
            
            if self.isEmergency1 != self.isEmergency2:
                response = "side_parking"
                message = self.TCP.encodeMsg(response)
                self.sendMsg(message)
                self.selectRoad("side_park")
                self.isEmergency2 = True
                print("emergency occured select road ", self.select_road)

            if self.ab_order == "stop":
                response = "stop"

            if self.isPowerOn == False:
                response = "wait"

            if response != self.curFlag:
                self.curFlag = response
                message = self.TCP.encodeMsg(response)
                self.sendMsg(message)
                # print(response)

            # self.count += 1
            # if self.count % 10 == 0:
            #     message = self.TCP.encodeMsg(response)
            #     self.sendMsg(message)
            
            self.updateUI(self.cls_list, self.direction, self.select_road)

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
    window.close_connection()
    window.db.closeLocal()
    
    # client_socket.close()