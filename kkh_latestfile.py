from ultralytics import YOLO
import cv2
import numpy as np

# load yolov8 model
model = YOLO('../../Desktop/object-tracking-yolov8-native-main/yolov8n.pt')

# load video
video_path = '../../Desktop/object-tracking-yolov8-native-main/test.mp4'
cap = cv2.VideoCapture(video_path)

ret = True
paused = False
mosaic_enabled = False  # 모자이크 기능 상태를 추적하는 변수


def apply_mosaic(image, x, y, w, h, block_size=10):
    face_roi = image[y:y + h, x:x + w]
    small = cv2.resize(face_roi, (w // block_size, h // block_size))
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y + h, x:x + w] = mosaic
    return image


# read frames
while ret:
    if not paused:
        ret, frame = cap.read()

    if ret:
        # detect objects
        # track objects
        results = model.track(frame, persist=True)

        # plot results
        frame_ = results[0].plot()

        # 탐지된 객체에 대해 처리
        if mosaic_enabled:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if model.names[cls] == 'person':
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # 얼굴 영역 추정 (상단 1/3)
                        face_h = int((y2 - y1) / 3)
                        frame_ = apply_mosaic(frame_, x1, y1, x2 - x1, face_h)

        # 모자이크 상태 표시
        status_text = "Mosaic: ON" if mosaic_enabled else "Mosaic: OFF"
        cv2.putText(frame_, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # visualize
        cv2.imshow('frame', frame_)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('p'):  # 'p' 키를 누르면 일시정지/재생 토글
            paused = not paused
        elif key == ord('m'):  # 'm' 키를 누르면 모자이크 on/off 토글
            mosaic_enabled = not mosaic_enabled
        elif key == 27:  # 'ESC' 키를 누르면 종료
            break
        elif key == ord('q'):  # 'q' 키를 누르면 종료 (기존 기능 유지)
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()