import cv2
import matplotlib.pyplot as plt
import dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model

IMG_HEIGHT = 224
IMG_WIDTH = 224
MARGIN = 10
def eye_rect_point(eye_lm):
    x1, y1 = np.amin(eye_lm, axis=0)
    x2, y2 = np.amax(eye_lm, axis=0)
    return (x1, y1), (x2, y2)

def crop_eye(frame, x1, y1, x2, y2):
    eye_width = x2 - x1
    eye_height = y2 - y1
    eye_x1 = int(x1 - MARGIN)
    eye_y1 = int(y1 - MARGIN)
    eye_x2 = int(x2 + MARGIN)
    eye_y2 = int(y2 + MARGIN)
    eye_x1 = max(eye_x1, 0)
    eye_y1 = max(eye_y1, 0)
    eye_x2 = min(eye_x2, frame.shape[1] - 1)
    eye_y2 = min(eye_y2, frame.shape[0] - 1)
    eye_image = frame[eye_y1:eye_y2, eye_x1:eye_x2]
    eye_image = cv2.resize(eye_image, dsize=(IMG_HEIGHT, IMG_WIDTH))
    eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    eye_image = np.repeat(eye_image[..., np.newaxis], 3, -1)
    eye_image = eye_image.reshape((-1, IMG_HEIGHT, IMG_WIDTH, 3))
    eye_image = eye_image / 255.
    return eye_image

model = load_model('c://ai_project01/eye_blink_mpdel/')
print("cnn model load")
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor("C:/ai_project01/eye_detect01/shape_predictor_68_face_landmarks.dat")

isSleep = [2.0 for _ in range(40)]
while cap.isOpened() == True:
    success, image = cap.read()

    if success == False:
        continue

    faces = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for face in faces:
        #print("얼굴 찾음!!")
        #print("얼굴 위치=", face)
        #print("="*100)
        lm = landmark_model(image, face)
        lm_arr = face_utils.shape_to_np(lm)
        #print("lm_arr=", lm_arr)

        (l_x1,l_y1), (l_x2, l_y2) = eye_rect_point(lm_arr[36:42])

        cv2.rectangle(image,
                      (l_x1, l_y1),
                      (l_x2, l_y2),
                      color=(0, 0, 255),
                      thickness=2
                      )
        (r_x1, r_y1), (r_x2, r_y2) = eye_rect_point(lm_arr[42:48])

        cv2.rectangle(image,
                      (r_x1, r_y1),
                      (r_x2, r_y2),
                      color=(0, 0, 255),
                      thickness=2
                      )

        eye_img_l = crop_eye(image, l_x1, l_y1, l_x2, l_y2)
        eye_img_r = crop_eye(image, r_x1, r_y1, r_x2, r_y2)
        if eye_img_l.size == 0:
            continue
        if eye_img_r.size == 0:
            continue

        pred_l = model.predict(eye_img_l)
        pred_r = model.predict(eye_img_r)
        state_l = f"{pred_l[0][0]:.1f}"
        state_r = f"{pred_r[0][0]:.1f}"

        print("-------ㅇㅇㅇㅇㅇㅇㅇㅇ----------------")
        isSleep.pop(0)
        isSleep.append(float(state_l) + float(state_r))
        if np.mean(isSleep) < 0.6:
            print("="*100)
            print("이 사람 잔다 - 점수 : ", np.mean(isSleep))
            print("="*100)
            cv2.putText(
                image,
                "SLEEPING.... z",
                (150, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.1,
                (255, 255, 255),
                6
            )
        else:
            print("점수 : ", np.mean(isSleep))




        cv2.putText(
            image,
            state_l,
            (l_x1, l_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            state_r,
            (r_x1, r_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        for p in lm.parts():
            cv2.circle(
                image, (p.x, p.y),
                radius=2,
                color=(255, 0, 0),
                thickness=2
            )
    cv2.imshow('webcam_window01', image)

    if cv2.waitKey(1) == ord('q'):
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #plt.imsave("cam_img.jpg", image)
        break

cap.release()