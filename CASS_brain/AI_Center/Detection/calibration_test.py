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
    'green_light' : 8.99
    #'obstacle' : 9.19,
}

# 카메라 캘리브레이션 값
mtx = np.array([[     672.52,           0,      330.84],
                [          0,      673.15,      257.85],
                [          0,           0,           1]])
                
dist = np.array([[   -0.44717,     0.63559,   0.0029907, -0.00055208,    -0.94232]])

# 설정값
KNOWN_WIDTH = 5.54  # cm, 피규어 실제 너비
KNOWN_DISTANCE = 29.9  # cm, 참조 거리

# Colors
BLACK = (0, 0, 0)

fonts = cv2.FONT_HERSHEY_COMPLEX

# focal length 계산 함수
def focal_length(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width

# 거리 계산 함수
def distance_finder(focal_length, real_width, width_in_frame):
    return (real_width * focal_length) / width_in_frame

# 바운딩 박스 폭 계산 함수
def pixel_width_data(results, image):
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        confidence = result.conf[0]
        class_id = int(result.cls[0])
        label = f"{names[class_id]}: {confidence:.2f}"

        # 바운딩 박스 및 레이블 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ref_image_width = x2 - x1
        print("폭:", ref_image_width)
        return ref_image_width
    return None

# 참조 이미지 처리
ref_image = cv2.imread("/home/yoon/ws/yolov8/data/data_dl/reference_img2.png")
ref_results = model.predict(ref_image, conf=0.45, verbose=False)

# 참조 이미지의 폭 및 초점 거리 계산
ref_image_width = pixel_width_data(ref_results, ref_image)
if ref_image_width:
    focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_width)
    print("초점 거리:", focal_length_found)
else:
    print("참조 이미지에서 객체를 탐지하지 못했습니다.")
    exit()

# 실시간 거리 측정
cap = cv2.VideoCapture('/home/yoon/ws/yolov8/data/video_file/test3.avi')

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
        
    width_in_frame = pixel_width_data(frame_results, calibrated_frame)

    # 거리 계산 및 표시
    if width_in_frame:
        distance = distance_finder(focal_length_found, KNOWN_WIDTH, width_in_frame)
        cv2.putText(
            calibrated_frame, f"Distance = {round(distance, 2)} cm", (50, 50), fonts, 1, BLACK, 2
        )

    # 결과 표시
    cv2.imshow("frame", calibrated_frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
