# 미러모드와 프레임 캡처 기능 추가

from ultralytics import YOLO
import cv2
import numpy as np
import time

# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
cap = cv2.VideoCapture(0)

# 전체화면 설정
cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 화면 크기 얻기
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret = True
paused = False
mosaic_enabled = False
blur_enabled = False
sharpen_enabled = False
grayscale_enabled = False
invert_enabled = False
mirror_enabled = False  # 새로운 변수 추가

# 기존의 함수들은 그대로 유지...

def apply_mirror(image):
    return cv2.flip(image, 1)
def apply_mosaic(image, x, y, w, h, block_size=10):
    face_roi = image[y:y + h, x:x + w]
    small = cv2.resize(face_roi, (w // block_size, h // block_size))
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y + h, x:x + w] = mosaic
    return image

def apply_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

def apply_sharpen(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_invert(image):
    return cv2.bitwise_not(image)


# read frames
while ret:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

    # 프레임을 화면 크기에 맞게 조정
    frame = cv2.resize(frame, (screen_width, screen_height))

    # detect objects
    results = model.track(frame, persist=True)

    # plot results
    frame_ = results[0].plot()

    if mosaic_enabled:
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if model.names[cls] == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_h = int((y2 - y1) / 3)
                    frame_ = apply_mosaic(frame_, x1, y1, x2 - x1, face_h)

    if blur_enabled:
        frame_ = apply_blur(frame_)

    if sharpen_enabled:
        frame_ = apply_sharpen(frame_)

    if grayscale_enabled:
        frame_ = apply_grayscale(frame_)
        if len(frame_.shape) == 2:  # 그레이스케일 이미지를 3채널로 변환
            frame_ = cv2.cvtColor(frame_, cv2.COLOR_GRAY2BGR)

    if invert_enabled:
        frame_ = apply_invert(frame_)

    if mirror_enabled:
        frame_ = apply_mirror(frame_)

    # 상태 표시
    status_text = []
    if mosaic_enabled: status_text.append("Mosaic: ON")
    if blur_enabled: status_text.append("Blur: ON")
    if sharpen_enabled: status_text.append("Sharpen: ON")
    if grayscale_enabled: status_text.append("Grayscale: ON")
    if invert_enabled: status_text.append("Invert: ON")
    if mirror_enabled: status_text.append("Mirror: ON")
    status = " Q ".join(status_text) if status_text else "All effects OFF"
    cv2.putText(frame_, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # visualize
    cv2.imshow('frame', frame_)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        paused = not paused
        print("Pause:", paused)
    elif key == ord('m'):
        mosaic_enabled = not mosaic_enabled
        print("Mosaic:", mosaic_enabled)
    elif key == ord('b'):
        blur_enabled = not blur_enabled
        print("Blur:", blur_enabled)
    elif key == ord('s'):
        sharpen_enabled = not sharpen_enabled
        print("Sharpen:", sharpen_enabled)
    elif key == ord('g'):
        grayscale_enabled = not grayscale_enabled
        print("Grayscale:", grayscale_enabled)
    elif key == ord('i'):
        invert_enabled = not invert_enabled
        print("Invert:", invert_enabled)
    elif key == ord('r'):
        mirror_enabled = not mirror_enabled
        print("Mirror:", mirror_enabled)
    elif key == ord('c'):
        # 프레임 캡처
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"capture_{timestamp}.jpg", frame_)
        print(f"Frame captured: capture_{timestamp}.jpg")
    elif key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()