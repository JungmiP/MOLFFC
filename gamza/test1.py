import cv2
import numpy as np
from exceptiongroup import catch
from ipykernel.pickleutil import class_type
from tensorflow import keras

import tensorflow as tf
from keras.models import load_model

# 환경 설정
IMG_HEIGHT = 224
IMG_WIDTH = 224

class_name = {
    0:'boring', 1:'focus_on', 2:'sleeping'
}



try:
    cnn_model = load_model('c://ai_project01/ab_result_Resnet')

except Exception as e:
    print(f"Error loading models: {e}")



# 첫 번째 웹캠 (ID 0)
cap1 = cv2.VideoCapture(0)

# # 두 번째 웹캠 (ID 1)
# cap2 = cv2.VideoCapture(1)

# 비디오 스트림이 열려 있는지 확인
if not cap1.isOpened() :
    print("카메라를 열 수 없습니다.")
    exit()

isFocused = [1 for _ in range(40)]
length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
print("frame_len", length)
while True:
    # 첫 번째 웹캠에서 프레임 읽기
    success1, image1 = cap1.read()
    #
    # # 두 번째 웹캠에서 프레임 읽기
    # success2, image2 = cap2.read()

    # 읽기가 성공했는지 확인
    if not success1 :
        print("카메라에서 프레임을 읽는 데 실패했습니다.")
        break

    # 이미지 좌우 대칭
    image1 = cv2.flip(image1, 1)


    # 이미지 크기 224, BGR2RGB 바꾸기, cnn_model.predict()
    if success1:
        # 이미지 전처리: 모델의 입력 크기와 형식에 맞게 변환
        # 1. 이미지 크기를 224x224로 리사이즈 (모델의 입력 크기에 맞게 수정)
        input_data = cv2.resize(image1, (224, 224))

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
        focusScore = np.mean(isFocused)
        print("focusScore=",focusScore)
        if focusScore < 0.8:
            cv2.putText(
                image1,
                "Boring",
                (150, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.1,
                (0, 0, 255),
                6
            )
            print("boring")
        elif focusScore < 1.5:
            cv2.putText(
                image1,
                "Focus On",
                (150, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.1,
                (0, 0, 255),
                6
            )
            print("focus on")
        else:
            cv2.putText(
                image1,
                "Sleeping",
                (150, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.1,
                (0, 0, 255),
                6
            )
            print("sleeping")


        prediction_text = class_name[predicted_class]

        # 예측 결과 출력
        # print("예측 결과: \n", predictions)
        # cv2.putText(image1, prediction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    else:
        print("웹캠에서 이미지를 읽어오지 못했습니다.")




    # 이어붙인 프레임을 하나의 창에 표시
    cv2.imshow('Combined Webcam', image1)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 리소스 해제
cap1.release()
#cap2.release()
cv2.destroyAllWindows()