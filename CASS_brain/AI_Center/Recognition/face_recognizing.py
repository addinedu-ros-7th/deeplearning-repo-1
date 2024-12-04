import face_recognition
import cv2
import numpy as np

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
my_image = face_recognition.load_image_file("./test/data/face/my_img/soyoung.png")
my_face_encoding = face_recognition.face_encodings(my_image)[0]

# Load a second sample picture and learn how to recognize it.
sb_image = face_recognition.load_image_file("./test/data/face/test_img/img_ms.jpg")
sb_face_encoding = face_recognition.face_encodings(sb_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    my_face_encoding,
    sb_face_encoding
]
known_face_names = [
    "so young",
    "min seop"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    if frame is not None:
        face_locations = face_recognition.face_locations(frame, model="cnn")
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.35)
            print(matches)
            name = "unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (200, 100, 5), 2)
            cv2.rectangle(frame, (left, bottom - 15), (right, bottom), (200, 100, 5), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, .5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

video_capture.release()
cv2.destroyAllWindows()