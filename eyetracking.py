import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# 마우스 감도
SMOOTHING_FACTOR = 0.3
last_x, last_y = screen_w // 2, screen_h // 2

# 눈 깜빡임 감지를 위한 설정
EYE_AR_THRESH = 0.3  # 눈 종횡비 임계값
BLINK_DURATION_THRESH = 0.1 # 깜빡임으로 인식할 시간
BLINK_INTERVAL_THRESH = 0.5  # 두 번의 깜빡임 사이의 최대 시간 간격
CLICK_COOLDOWN = 1.0  # 클릭 간 최소 시간 간격

# 깜빡임 감지 변수
blink_timestamps = []
last_click_time = time.time()
blink_start_time = None

# 눈의 종횡비
def calculate_ear(eye_points, landmarks):
    points = np.array([[landmarks[point].x, landmarks[point].y] for point in eye_points])
    
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    
    horizontal = np.linalg.norm(points[0] - points[3])
    
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear

def get_eye_position(landmarks, frame):
    left_eye_points = [(landmarks[p].x, landmarks[p].y) for p in [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385]]
    right_eye_points = [(landmarks[p].x, landmarks[p].y) for p in [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160]]
    
    # 양쪽 눈의 중심점
    left_eye = np.mean(left_eye_points, axis=0)
    right_eye = np.mean(right_eye_points, axis=0)
    
    # 전체 눈 위치의 중심점
    eye_pos = np.mean([left_eye, right_eye], axis=0)
    
    # 화면 좌표로 변환
    x = int(eye_pos[0] * screen_w)
    y = int(eye_pos[1] * screen_h)
    
    return x, y

def detect_blink(landmarks):
    left_eye = [362, 385, 387, 263, 373, 380]
    right_eye = [33, 160, 158, 133, 153, 144]
    
    left_ear = calculate_ear(left_eye, landmarks)
    right_ear = calculate_ear(right_eye, landmarks)
    
    ear = (left_ear + right_ear) / 2.0
    
    return ear < EYE_AR_THRESH

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # 눈 위치 추적 및 커서 이동
            x, y = get_eye_position(landmarks, frame)
            smooth_x = int(last_x + (x - last_x) * SMOOTHING_FACTOR)
            smooth_y = int(last_y + (y - last_y) * SMOOTHING_FACTOR)
            pyautogui.moveTo(smooth_x, smooth_y)
            last_x, last_y = smooth_x, smooth_y

            # 눈 깜빡임 감지 및 더블 클릭 처리
            current_time = time.time()
            is_blink = detect_blink(landmarks)

            if is_blink:
                if blink_start_time is None:
                    blink_start_time = current_time
            elif not is_blink and blink_start_time is not None:
                # 눈 뜸 - 깜빡임 종료
                blink_end_time = current_time
                blink_duration = blink_end_time - blink_start_time
                if blink_duration >= BLINK_DURATION_THRESH: # 최소 깜빡임 시간 충족
                    blink_timestamps.append(blink_end_time)
                blink_start_time = None

            # 클릭 처리
            if len(blink_timestamps) >= 2:
                if blink_timestamps[-1] - blink_timestamps[-2] < BLINK_INTERVAL_THRESH:
                    if current_time - last_click_time >= CLICK_COOLDOWN:
                        pyautogui.click()
                        print("두 번 깜빡임. 클릭!")
                        last_click_time = current_time
                blink_timestamps = []  # 클릭 후 리스트 초기화


        cv2.imshow('커서 컨트롤', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()