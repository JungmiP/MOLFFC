import cv2
import dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
from l2cs import Pipeline, render
import torch
from datetime import datetime

# 10분 동안의 집중도 기록을 저장할 리스트
focus_scores = []
my_score = []  # 5분 평균을 저장하는 리스트

IMG_HEIGHT = 224
IMG_WIDTH = 224
MARGIN = 10

class_name = {
    0:'boring', 1:'focus_on', 2:'sleeping'
}

# 경고 및 성공 메시지를 위한 시간 기록 변수
message_start_time = None  # 메시지 시작 시간을 기록
gauge_display_time = None  # 게이지 시작 시간을 기록
display_message = None  # 현재 띄울 메시지
initial_check_done = False  # 첫 1분 측정 여부
first_1min_done = False  # 첫 1분이 끝났는지 여부
gauge_displayed = False  # 게이지가 화면에 표시 중인지 여부

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

# 깜빡임 모델 로드
blink_model = load_model('c://ai_project01/eye_blink_mpdel/')

# 시선 추적 모델 로드
gaze_pipeline = Pipeline(
    weights="C://ai_project01/eye_detect01/L2CSNet_gaze360.pkl",
    arch="ResNet50",
    device=torch.device('cuda')
)

cnn_model = load_model('c://ai_project01/ab_result_Resnet')


# 카메라 및 얼굴 검출 모델 설정
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor("C:/ai_project01/eye_detect01/shape_predictor_68_face_landmarks.dat")

isSleep = [2.0 for _ in range(40)]  # 최근 40번의 깜빡임 상태를 저장
isFocused = [1 for _ in range(40)]
faceScore = sleepingScore = 0
# 시작 시간
start_time = datetime.now()

while cap.isOpened():
    current_time = datetime.now()
    elapsed_time = (current_time - start_time).total_seconds()
    print(elapsed_time)

    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)

    if success:
        # 이미지 전처리: 모델의 입력 크기와 형식에 맞게 변환
        # 1. 이미지 크기를 224x224로 리사이즈 (모델의 입력 크기에 맞게 수정)
        input_data = cv2.resize(image, (224, 224))

        # 2. BGR에서 RGB로 변환 (cv2는 기본적으로 BGR 형식이므로 변환 필요)
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)

        # 3. 모델이 요구하는 차원 추가 (배치 크기 추가) => (1, 224, 224, 3)
        input_data = np.expand_dims(input_data, axis=0)

        # 4. 이미지 데이터를 float32 타입으로 변환 후 스케일링 (0~1 사이 값으로)
        input_data = input_data.astype('float32') / 255.0

        # 모델 예측
        predictions = cnn_model.predict(input_data)

        # 예측값을 문자열로 변환 (가장 높은 확률을 가진 클래스 출력)
        predicted_class = np.argmax(predictions, axis=1)[0]
        print("predicted_class : ", predicted_class)

        isFocused.pop(0)
        isFocused.append(int(predicted_class))
        #print("np.mean(isFocused)", np.mean(isFocused))
        faceScore = 1 if 0.5 < np.mean(isFocused) < 1.5 else 0


    faces = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for face in faces:
        # 눈의 위치 및 깜빡임 상태 등 기존 코드 그대로 처리
        lm = landmark_model(image, face)
        lm_arr = face_utils.shape_to_np(lm)
        (l_x1, l_y1), (l_x2, l_y2) = eye_rect_point(lm_arr[36:42])
        (r_x1, r_y1), (r_x2, r_y2) = eye_rect_point(lm_arr[42:48])

        eye_img_l = crop_eye(image, l_x1, l_y1, l_x2, l_y2)
        eye_img_r = crop_eye(image, r_x1, r_y1, r_x2, r_y2)

        if eye_img_l.size == 0 or eye_img_r.size == 0:
            continue

        pred_l = blink_model.predict(eye_img_l)
        pred_r = blink_model.predict(eye_img_r)
        state_l = f"{pred_l[0][0]:.1f}"
        state_r = f"{pred_r[0][0]:.1f}"

        isSleep.pop(0)
        isSleep.append(float(state_l) + float(state_r))

        sleepingScore = np.mean(isSleep) / 2

        # if sleepingScore < 0.5:
            # cv2.putText(
            #     image,
            #     "SLEEPING.... z",
            #     (150, 150),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     2.1,
            #     (255, 255, 255),
            #     6
            # )

        cv2.putText(image, state_l, (l_x1, l_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, state_r, (r_x1, r_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        results = gaze_pipeline.step(image)
        image = render(image, results)

        # 주시 상태에 따른 집중도 업데이트
        for i in range(results.pitch.shape[0]):
            bbox = results.bboxes[i]
            pitch = results.pitch[i]
            yaw = results.yaw[i]

            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[2])
            y_max = int(bbox[3])

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            center_x = int(x_min + bbox_width / 2.0)
            center_y = int(y_min + bbox_height / 2.0)

            dx = -bbox_width * np.sin(pitch) * np.cos(yaw)
            dy = -bbox_width * np.sin(yaw)

            focus_width = bbox_width / 3
            focus_height = 200

            gaze_point = tuple(np.round([center_x + dx, center_y + dy]).astype(int))

            quadrants = [
                ("center", (int(center_x - focus_width), int(center_y - focus_height / 2),
                            int(center_x + focus_width), int(center_y + focus_height / 2))),
                ("top_left", (x_min, y_min, center_x, center_y)),
                ("top_right", (center_x, y_min, x_max, center_y)),
                ("bottom_left", (x_min, center_y, center_x, y_max)),
                ("bottom_right", (center_x, center_y, x_max, y_max)),
            ]

            focus_level = "boring"
            for quadrant, (qx_min, qy_min, qx_max, qy_max) in quadrants:
                if qx_min <= gaze_point[0] <= qx_max and qy_min <= gaze_point[1] <= qy_max:
                    #cv2.rectangle(image, (qx_min, qy_min), (qx_max, qy_max), (255, 255, 255), 2)

                    if quadrant == "center":
                        focus_level = "focus"
                        cv2.putText(image, "focus", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                    else:
                        cv2.putText(image, "boring", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                    break
            focusScore = 1 if focus_level == "focus" else 0.1
            focus_scores.append(sleepingScore * focusScore * faceScore)
            #print("sleepingScore", sleepingScore)
            #print("focusScore", focusScore)
            #print("faceScore", faceScore)
            print("focus_scores", np.mean(focus_scores) * 100)
    # 첫 1분 동안의 집중도를 계산
    if not first_1min_done and elapsed_time >= 30:  # 처음 1분
        if focus_scores:
            avg_focus_1min = np.mean(focus_scores)
            focus_scores = []  # 1분 이후에는 5분을 측정하므로 초기화
            first_1min_done = True  # 첫 1분 측정 완료

            # 첫 1분 메시지 설정
            if avg_focus_1min < 0.5:
                display_message = "Warning: You need to focus! Please pay attention."
            else:
                display_message = "Perfect! You are focusing well."

            message_start_time = current_time  # 메시지 시작 시간 기록

    # 첫 1분 이후, 5분마다 집중도 계산
    if first_1min_done and (elapsed_time % 60 <= 1):  # 5분 == 300초
        if focus_scores:
            avg_focus_5min = np.mean(focus_scores)  # 5분 동안의 평균 계산
            focus_scores = []  # 다음 5분을 위해 리스트 초기화

            # 메시지를 결정 (평균 집중도가 50% 미만인지 초과인지에 따라)
            if avg_focus_5min < 50:
                display_message = "Warning: You need to focus! Please pay attention."
            else:
                display_message = "Perfect! You are focusing well."

            message_start_time = current_time  # 메시지를 띄운 시간을 기록

            # 5분 집중도 결과를 my_score 리스트에 저장
            my_score.append(avg_focus_5min)

            # 게이지 시작 시간을 기록
            gauge_display_time = current_time
            gauge_displayed = True

    # 메시지를 30초 동안 유지
    # if message_start_time and (current_time - message_start_time).total_seconds() <= 30:
        # cv2.putText(image, display_message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
        #             (0, 255, 0) if "Perfect" in display_message else (0, 0, 255), 3)

    # 게이지를 20초 동안 유지
    if gauge_displayed and (current_time - gauge_display_time).total_seconds() <= 20:
        if len(my_score) > 0:
            avg_focus = my_score[-1]  # 마지막 5분 평균 집중도
            screen_width = image.shape[1]
            gauge_width = int(avg_focus * screen_width)  # 평균 집중도에 비례한 게이지 너비
            cv2.rectangle(image, (50, 400), (50 + gauge_width, 450), (0, 255, 0), -1)
            cv2.putText(image, f"Avg Focus: {avg_focus * 100 :.2f}", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        gauge_displayed = False  # 20초가 지나면 게이지 사라짐

    cv2.imshow('webcam_window01', image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
