import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('/home/yoon/ws/yolov8/train5/weights/best.pt')
names = model.model.names

# 실제 객체의 폭 (cm)
known_widths = {
    'person' : 5.54,
    'red_light' : 8.99,
    'green_light' : 8.99,
    'goat' : 9.89
}

# 카메라 캘리브레이션 값
mtx = np.array([[     672.52,           0,      330.84],
                [          0,      673.15,      257.85],
                [          0,           0,           1]])
                
dist = np.array([[   -0.44717,     0.63559,   0.0029907, -0.00055208,    -0.94232]])

# 설정값
KNOWN_WIDTH = 5.54  # cm, 피규어 실제 너비
KNOWN_DISTANCE = 29.9  # cm, 참조 거리


# focal length 계산 함수
def focal_length(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width

# 거리 계산 함수
def distance_finder(focal_length, real_width, width_in_frame):
    return (real_width * focal_length) / width_in_frame

def color_finder(name):
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

# 바운딩 박스 폭 계산 함수
def pixel_width_data(results, image):
    class_names = []
    widths = []
    boxes = []  # 추가: 각 바운딩 박스 좌표 저장
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        confidence = result.conf[0]
        class_id = int(result.cls[0])
        label = f"{names[class_id]}: {confidence:.2f}"

        # 바운딩 박스 및 레이블 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), color_finder(names[class_id]), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_finder(names[class_id]), 2)

        ref_image_width = x2 - x1
        #print("폭:", ref_image_width)
        class_names.append(names[class_id])
        widths.append(ref_image_width)
        boxes.append((x1, y1, x2, y2))  # 바운딩 박스 좌표 저장

    return class_names, widths, boxes  # boxes도 반환


# 참조 이미지 처리
ref_image = cv2.imread("/home/yoon/ws/yolov8/data/data_dl/reference_img2.png")
ref_results = model.predict(ref_image, conf=0.45, verbose=False)

# 참조 이미지의 폭 및 초점 거리 계산
ref_names, ref_widths, ref_boxes = pixel_width_data(ref_results, ref_image)
if ref_widths[0]:
    focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_widths[0])
    print("초점 거리:", focal_length_found)
else:
    print("참조 이미지에서 객체를 탐지하지 못했습니다.")
    exit()

# 실시간 거리 측정
#cap = cv2.VideoCapture('/home/yoon/ws/yolov8/data/video_file/test3.avi')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    h, w = frame.shape[:2]
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 왜곡 계수 보정
    calibrated_frame = cv2.undistort(frame, mtx, dist, new_matrix)

    # YOLO 예측
    frame_results = model.predict(calibrated_frame, conf=0.55, verbose=False)
        
    class_names, widths, boxes = pixel_width_data(frame_results, calibrated_frame)

    for name, width, (x1, y1, x2, y2) in zip(class_names, widths, boxes):
        if name in known_widths:  # 초록불, 빨간불, 사람
            # 자동차 길이 빼서 보정
            distance = distance_finder(focal_length_found, known_widths[name], width) - 16
            cv2.putText(calibrated_frame, f"{round(distance, 2)} cm", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_finder(name), 2)

    # 결과 표시
    cv2.imshow("frame", calibrated_frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
