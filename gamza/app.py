from flask import Flask
from flask import request
import base64
import json

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
total_score = []

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

detector = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor("C:/ai_project01/eye_detect01/shape_predictor_68_face_landmarks.dat")

isSleep = [2.0 for _ in range(40)]  # 최근 40번의 깜빡임 상태를 저장
isFocused = [1 for _ in range(40)]
faceScore = sleepingScore = 0
# 시작 시간
start_time = datetime.now()


app = Flask(__name__)

# 1초마다 이미지를 받아와 점수 계산하는 함수
@app.route('/calScore', methods=['POST'])
def calScore():
    sleepingScore = faceScore = focusScore = 0
    json_image = request.get_json()
    encoded_data_arr = json_image.get("data")
    #for index, encoded_data in enumerate(encoded_data_arr):

    encoded_data = encoded_data_arr.replace("data:image/jpeg;base64,", "")

    decoded_data = base64.b64decode(encoded_data)
    nparr = np.fromstring(decoded_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image = cv2.flip(image, 1)

    input_data = cv2.resize(image, (224, 224))

    # 2. BGR에서 RGB로 변환 (cv2는 기본적으로 BGR 형식이므로 변환 필요)
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)

    # 3. 모델이 요구하는 차원 추가 (배치 크기 추가) => (1, 224, 224, 3)
    input_data = np.expand_dims(input_data, axis=0)

    # 4. 이미지 데이터를 float32 타입으로 변환 후 스케일링 (0~1 사이 값으로)
    input_data = input_data.astype('float32') / 255.0

    # 표정 분류 모델 적용
    # 모델 예측
    predictions = cnn_model.predict(input_data)

    # 예측값을 문자열로 변환 (가장 높은 확률을 가진 클래스 출력)
    predicted_class = np.argmax(predictions, axis=1)[0]
    print("predicted_class : ", predicted_class)

    isFocused.pop(0)
    isFocused.append(int(predicted_class))
    # print("np.mean(isFocused)", np.mean(isFocused))
    # 표정 분류 점수
    faceScore = 1 if 0.5 < np.mean(isFocused) < 1.5 else 0.1



    # 얼굴 탐지하여 눈 위치 데이터 얻기
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

        # 눈 깜빡임 감지
        pred_l = blink_model.predict(eye_img_l)
        pred_r = blink_model.predict(eye_img_r)
        state_l = f"{pred_l[0][0]:.1f}"
        state_r = f"{pred_r[0][0]:.1f}"

        # 졸림 여부 점수로 환산
        isSleep.pop(0)
        isSleep.append(float(state_l) + float(state_r))

        sleepingScore = np.mean(isSleep) / 2

    # 시선 방향 탐지

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
                if quadrant == "center":
                    focus_level = "focus"
        # 시선 방향으로 집중도 계산

        focusScore = 1 if focus_level == "focus" else 0.1
        print("focusScore=", focusScore)
        print("faceScore=", faceScore)
        print("sleppingScore=", sleepingScore)
        focus_scores.append(sleepingScore * focusScore * faceScore)

        print("focus_scores", focus_scores[-1])
        if len(focus_scores) >= 300:
            focus_scores.pop(0)
        print(focus_scores)
    return str(focus_scores[-1])


    #5분마다 점수 종합하여 send
@app.route('/sendScore', methods=["POST"])
def sendScore():
    total_score.append(np.mean(focus_scores)*100)

    return "최근 5분간의 점수는" + total_score[-1]

if __name__ == '__main__':
    app.run()