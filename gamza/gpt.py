import cv2
import dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
from l2cs import Pipeline, render
import torch

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
MARGIN = 10
SLEEP_THRESHOLD = 0.6
BORED_THRESHOLD = 0.6
SLEEP_DURATION = 10
BORED_DURATION = 10


# Eye detection functions
def eye_rect_point(eye_lm):
    x1, y1 = np.amin(eye_lm, axis=0)
    x2, y2 = np.amax(eye_lm, axis=0)
    return (x1, y1), (x2, y2)


def crop_eye(frame, x1, y1, x2, y2):
    eye_x1 = max(int(x1 - MARGIN), 0)
    eye_y1 = max(int(y1 - MARGIN), 0)
    eye_x2 = min(int(x2 + MARGIN), frame.shape[1] - 1)
    eye_y2 = min(int(y2 + MARGIN), frame.shape[0] - 1)

    eye_image = frame[eye_y1:eye_y2, eye_x1:eye_x2]
    eye_image = cv2.resize(eye_image, dsize=(IMG_HEIGHT, IMG_WIDTH))
    eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    eye_image = np.repeat(eye_image[..., np.newaxis], 3, -1)
    eye_image = eye_image.reshape((-1, IMG_HEIGHT, IMG_WIDTH, 3)) / 255.0
    return eye_image


# Load models
blink_model = load_model('c://ai_project01/eye_blink_model/')
gaze_pipeline = Pipeline(weights="C://ai_project01/eye_detect01/L2CSNet_gaze360.pkl", arch="ResNet50",
                         device=torch.device('cuda'))
detector = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor("C://ai_project01/eye_detect01/shape_predictor_68_face_landmarks.dat")

# Variables for sleep and boredom detection
is_sleeping = 0
is_bored = 0
is_sleep = [2.0 for _ in range(40)]
is_bored_list = [0 for _ in range(40)]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    faces = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for face in faces:
        lm = landmark_model(image, face)
        lm_arr = face_utils.shape_to_np(lm)

        # Get eye positions
        (l_x1, l_y1), (l_x2, l_y2) = eye_rect_point(lm_arr[36:42])
        (r_x1, r_y1), (r_x2, r_y2) = eye_rect_point(lm_arr[42:48])

        # Crop eyes and predict blink state
        eye_img_l = crop_eye(image, l_x1, l_y1, l_x2, l_y2)
        eye_img_r = crop_eye(image, r_x1, r_y1, r_x2, r_y2)

        if eye_img_l.size == 0 or eye_img_r.size == 0:
            continue

        pred_l = blink_model.predict(eye_img_l)
        pred_r = blink_model.predict(eye_img_r)
        state_l = float(pred_l[0][0])
        state_r = float(pred_r[0][0])

        # Update sleep and boredom tracking
        is_sleep.pop(0)
        is_sleep.append(state_l + state_r)

        if np.mean(is_sleep) < SLEEP_THRESHOLD:
            is_sleeping += 1
            if is_sleeping > SLEEP_DURATION * 30:  # Assuming 30 FPS
                cv2.putText(image, "집중필요!", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.1, (255, 255, 255), 6)
        else:
            is_sleeping = 0

        # Boredom detection
        if np.mean(is_sleep) > BORED_THRESHOLD:
            is_bored += 1
            if is_bored > BORED_DURATION * 30:  # Assuming 30 FPS
                cv2.putText(image, "집중필요!", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.1, (255, 255, 255), 6)
        else:
            is_bored = 0

        # Gaze estimation
        results = gaze_pipeline.step(image)
        image = render(image, results)

        # Draw rectangles and labels
        cv2.rectangle(image, (l_x1, l_y1), (l_x2, l_y2), (0, 0, 255), 2)
        cv2.rectangle(image, (r_x1, r_y1), (r_x2, r_y2), (0, 0, 255), 2)
        cv2.putText(image, f"Left: {state_l:.1f}", (l_x1, l_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Right: {state_r:.1f}", (r_x1, r_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    2)

    cv2.imshow('webcam_window01', image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
