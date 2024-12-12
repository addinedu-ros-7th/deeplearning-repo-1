import warnings  # 경고 메시지를 제어하기 위한 표준 라이브러리
warnings.filterwarnings("ignore", category=DeprecationWarning)  # DeprecationWarning 경고를 무시하도록 설정

import sys  # 시스템 관련 기능(인자, 종료 등)을 제공하는 표준 라이브러리
import time  # 시간 관련 작업(딜레이, 현재 시간 측정)을 위한 라이브러리
import cv2  # OpenCV 라이브러리 (컴퓨터 비전 처리에 사용)
from PyQt5.QtWidgets import *  # PyQt5의 UI 위젯 관련 클래스
from PyQt5.QtGui import *  # PyQt5의 그래픽 처리 관련 클래스
from PyQt5.QtCore import *  # PyQt5의 코어 기능(QThread 등)
from PyQt5 import uic  # PyQt5 UI 파일(.ui) 로더
# from Modules import *  # 사용자 정의 모듈 가져오기
import sympy

# .ui 파일로부터 Python 클래스 로드
from_class = uic.loadUiType("../dev_ws/deeplearning-repo-1/CASS_brain/GUI/CASS_ui.ui")[0]


# CASS UI 클래스 정의
class CASS_ui(QMainWindow, from_class):
    def __init__(self):
        super().__init__()  # 부모 클래스(QDialog) 초기화
        self.setupUi(self)  # .ui 파일로부터 UI 초기화
        
        # ui update시 필요한 변수
        self.input_data = {'objects':[], 'direction':None, 'select_road':None}
        self.UI_objs = {'person':'dynamic', 
                        'obstacle':'static', 
                        'goat':'dynamic',
                        'red_light':'red',
                        'green_light':'green'}
        self.objs_keys = self.UI_objs.keys()

        # # UI 상태 화면 초기화
        self.labelTrafficLight_R.hide()
        self.labelTrafficLight_G.hide()

        self.labelDynamicObstacle_ON.hide()
        self.labelStaticObstacle_ON.hide()

        self.labelDirection_back.hide()
        self.labelDirection_straight.hide()
        self.labelDirection_left.hide()
        self.labelDirection_right.hide()

        self.labelSelectRoad_left.hide()
        self.labelSelectRoad_right.hide()

    
    def push_info(self):
        objs, direction, select_road = self.input_data.values()
        self.updateUI(objs, direction, select_road)


    def updateUI(self, objs, direction, select_road):
        objs = set(objs)
        objs = [self.UI_objs[obj] for obj in objs if obj in self.objs_keys]
        if 'dynamic' in objs:
            self.labelDynamicObstacle_ON.show()
        else:
            self.labelDynamicObstacle_ON.hide()
        if 'static' in objs:
            self.labelStaticObstacle_ON.show()
        else:
            self.labelStaticObstacle_ON.hide()
        if 'red' in objs:
            self.labelTrafficLight_R.show()
        else:
            self.labelTrafficLight_R.hide()
        if 'green' in objs:
            self.labelTrafficLight_G.show()
        else:
            self.labelTrafficLight_G.hide()

        if 'straight' == direction:
            self.labelDirection_straight.show()
        else:
            self.labelDirection_straight.hide()

        if 'left' == direction:
            self.labelDirection_left.show()
        else:
            self.labelDirection_left.hide()
        if 'right' == direction:
            self.labelDirection_right.show()
        else:
            self.labelDirection_right.hide()
        if 'reverse' == direction:
            self.labelDirection_back.show()
        else:
            self.labelDirection_back.hide()

        if 'left' == select_road:
            self.labelSelectRoad_left.show()
        else:
            self.labelSelectRoad_left.hide()
        if 'right' == select_road:
            self.labelSelectRoad_right.show()
        else:
            self.labelSelectRoad_right.hide()


if __name__ == "__main__":  # 메인 실행부
    app = QApplication(sys.argv)  # PyQt5 애플리케이션 생성
    window = CASS_ui()  # CASS UI 인스턴스 생성
    window.show()  # UI 표시
    app.exec_()  # 이벤트 루프 실행
