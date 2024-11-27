import cv2
import dlib
import numpy as np
from imutils import face_utils

from l2cs import Pipeline, render
import torch


IMG_HEIGHT =224
IMG_WIDTH = 224
MARGIN = 10


def eye_rect_point(eye_lm):
    x1,y1 = np.amin(eye_lm, axis=0)
    x2,y2 = np.amax(eye_lm ,axis=0)
    return (x1,y1),(x2,y2)

def crop_eye(frame, x1,y1,x2,y2):
    eye_width = x2 - x1
    eye_height = y2 - y1

    eye_x1 = int(x1 - MARGIN)
    eye_y1 = int(y1 - MARGIN)

    eye_x2 = int(x2 - MARGIN)
    eye_y2 = int(y2 - MARGIN)

    eye_x1 = max(eye_x1, 0)
    eye_y1 = max(eye_y1, 0)

    eye_x2 = min(eye_x2 , frame.shape[1]-1)

    eye_y2 = min(eye_y2, frame.shape[0]-1)

    eye_image = frame[eye_y1: eye_y2 , eye_x1:eye_x2]
    eye_image = cv2.resize(eye_image, dsize=(IMG_HEIGHT, IMG_WIDTH))

    eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    eye_image = np.repeat( eye_image [..., np.newaxis], 3,-1)
    eye_image = eye_image.reshape((-1, IMG_HEIGHT, IMG_WIDTH,3))

    eye_image = eye_image.reshape((-1, IMG_HEIGHT, IMG_WIDTH, 3))
    eye_image = eye_image / 255.
    return eye_image

# def gaze_direction(pitch, yaw):
#     # 시선 방향에 따라 집중도를 판단
#     if -15 < pitch < 15 and -15 < yaw < 15:
#         return 0  # 집중
#     else:
#     # elif (pitch < -15 and -45 < yaw < 45) or (pitch > 15 and -45 < yaw < 45):
#         return 1  # 지루함

cap = cv2.VideoCapture(0)

gaze_pipeline = Pipeline(
    weights="C://ai_project01/eye_detect01/L2CSNet_gaze360.pkl",
    arch="ResNet50",
    device=torch.device('cuda')
)

detector = dlib.get_frontal_face_detector()

landmark_model = dlib.shape_predictor("C://ai_project01/eye_detect01/shape_predictor_68_face_landmarks.dat")

while cap.isOpened()==True:

    success, image = cap.read()

    if success ==False :
        continue


    image = cv2.flip(image, 1)

    faces = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for face in faces:

        lm = landmark_model(image, face)
        lm_arr = face_utils.shape_to_np(lm)
        (l_x1, l_y1), (l_x2, l_y2) = eye_rect_point(lm_arr[36:42])
        (r_x1, r_y1), (r_x2, r_y2) = eye_rect_point(lm_arr[42:48])

        eye_img_l = crop_eye(image, l_x1, l_y1, l_x2, l_y2)
        eye_img_r = crop_eye(image, r_x1, r_y1, r_x2, r_y2)

        if eye_img_l.size == 0 or eye_img_r.size == 0:
            cv2.putText(image, "슬리핑", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if eye_img_l.size ==0:
            continue
        if eye_img_r.size == 0:
            continue

        results = gaze_pipeline.step(image)
        # print("="*100)
        # print("results=", results)
        # print("="*100)
        image = render(image, results)
        for i in range(results.pitch.shape[0]):
            bbox = results.bboxes[i]
            pitch = results.pitch[i]
            yaw = results.yaw[i]

            # focusScore = gaze_direction(pitch, yaw)

            x_min = int(bbox[0])
            if x_min < 0:
                x_min = 0

            y_min = int(bbox[1])
            if y_min < 0:
                y_min = 0


            x_max = int(bbox[2])
            y_max = int(bbox[3])

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            label = "yaw {:.2f} pitch {:.2f}".format(
                 yaw / np.pi * 180, pitch / np.pi * 180
            )

            cv2.putText(
                image,
                label,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255,0,0),
                2
            )

            # cv2.putText(
            #     image,
            #     "focus" if focusScore == 0 else "boring",
            #     (x_min + 10, y_min - 10),
            #     cv2.FONT_HERSHEY_PLAIN,
            #     1,
            #     (255, 0, 0),
            #     2
            # )

            center_x = int(x_min + bbox_width /2.0)
            center_y = int(y_min + bbox_height /2.0)

            dx = -bbox_width * np.sin(pitch) * np.cos(yaw)
            dy = -bbox_width * np.sin(yaw)
            focus_height = 200
            focus_width = bbox_width / 3
            quadrants = [
                ("center",
                 # (int(center_x - bbox_width /10), int(center_y - bbox_height /10),
                 #  int(center_x + bbox_width /10), int (center_y + bbox_height /10))),
                 (int(center_x - focus_width), int(center_y - focus_height / 2),  # 상단 100px 위
                  int(center_x + focus_width), int(center_y + focus_height / 2))),
                ("top_left", (x_min, y_min, center_x, center_y)),
                ("top_right", (center_x, y_min, x_max, center_y)),
                ("bottom_left", (x_min, center_y, center_x, y_max)),
                ("bottom_right", (center_x, center_y, x_max, y_max)),

            ]

            gaze_point = tuple(np.round([center_x + dx, center_y + dy]).astype(int))

            for quadrant, (qx_min, qy_min, qx_max, qy_max) in quadrants:

                if qx_min <= gaze_point[0] <= qx_max and qy_min <= gaze_point[1] <= qy_max:
                    cv2.rectangle(image,
                                  (qx_min, qy_min),
                                  (qx_max, qy_max),
                                  (255,255,255),
                                  2)

                    cv2.putText(image,
                                quadrant,
                                (0,50),
                                cv2.FONT_HERSHEY_PLAIN,
                                2,
                                (255,0,0),
                                2)

                    # if quadrant == "center":
                    #     cv2.putText(image, "focus", (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    # else:
                    #     cv2.putText(image, "boring", (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

                    break

    cv2.imshow('webcam_window01', image)

    if cv2.waitKey(1) == ord('q'):
       break

cap.release()