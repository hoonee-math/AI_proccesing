from ultralytics import YOLO
import cv2
import numpy as np

# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = './test.mp4'
cap = cv2.VideoCapture(video_path)

ret = True
paused = False
mosaic_enabled = False
blur_enabled = False
sharpen_enabled = False

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

# read frames
while ret:
    if not paused:
        ret, frame = cap.read()

    if ret:
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

        # 상태 표시
        status_text = []
        if mosaic_enabled: status_text.append("Mosaic: ON")
        if blur_enabled: status_text.append("Blur: ON")
        if sharpen_enabled: status_text.append("Sharpen: ON")
        status = " | ".join(status_text) if status_text else "All effects OFF"
        cv2.putText(frame_, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # visualize
        cv2.imshow('frame', frame_)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('p'):
            paused = not paused
        elif key == ord('m'):
            mosaic_enabled = not mosaic_enabled
        elif key == ord('b'):
            blur_enabled = not blur_enabled
        elif key == ord('s'):
            sharpen_enabled = not sharpen_enabled
        elif key == 27 or key == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()