import cv2
import numpy as np
from ultralytics import YOLO
import time

class YoloProcessor:
    def __init__(self, model_path='yolov8n.pt'):
        # YOLO 모델 초기화
        self.model = YOLO(model_path)
        # 각종 효과 상태 초기화
        self.paused = False
        self.mosaic_enabled = False
        self.blur_enabled = False
        self.sharpen_enabled = False
        self.grayscale_enabled = False
        self.invert_enabled = False
        self.mirror_enabled = False

    def process_frame(self, frame):
        # YOLO 객체 검출 수행
        results = self.model.track(frame, persist=True)
        frame_ = results[0].plot()

        # 각 효과 적용
        if self.mosaic_enabled:
            frame_ = self.apply_mosaic(frame_, results)
        if self.blur_enabled:
            frame_ = self.apply_blur(frame_)
        if self.sharpen_enabled:
            frame_ = self.apply_sharpen(frame_)
        if self.grayscale_enabled:
            frame_ = self.apply_grayscale(frame_)
        if self.invert_enabled:
            frame_ = self.apply_invert(frame_)
        if self.mirror_enabled:
            frame_ = self.apply_mirror(frame_)

        return frame_

    def apply_mosaic(self, image, results):
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if self.model.names[cls] == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_h = int((y2 - y1) / 3)
                    face_roi = image[y1:y1+face_h, x1:x2]
                    block_size = 10
                    small = cv2.resize(face_roi, (x2-x1 // block_size, face_h // block_size))
                    mosaic = cv2.resize(small, (x2-x1, face_h), interpolation=cv2.INTER_NEAREST)
                    image[y1:y1+face_h, x1:x2] = mosaic
        return image

    def apply_blur(self, image):
        return cv2.GaussianBlur(image, (15, 15), 0)

    def apply_sharpen(self, image):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)

    def apply_grayscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def apply_invert(self, image):
        return cv2.bitwise_not(image)

    def apply_mirror(self, image):
        return cv2.flip(image, 1)

    def capture_frame(self, frame):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"capture_{timestamp}.jpg", frame)
        print(f"Frame captured: capture_{timestamp}.jpg")

    def get_status_text(self):
        status_text = []
        if self.mosaic_enabled: status_text.append("Mosaic: ON")
        if self.blur_enabled: status_text.append("Blur: ON")
        if self.sharpen_enabled: status_text.append("Sharpen: ON")
        if self.grayscale_enabled: status_text.append("Grayscale: ON")
        if self.invert_enabled: status_text.append("Invert: ON")
        if self.mirror_enabled: status_text.append("Mirror: ON")
        return status_text

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused

    # 각 효과의 토글 메서드
    def toggle_mosaic(self):
        self.mosaic_enabled = not self.mosaic_enabled
        return self.mosaic_enabled

    def toggle_blur(self):
        self.blur_enabled = not self.blur_enabled
        return self.blur_enabled

    def toggle_sharpen(self):
        self.sharpen_enabled = not self.sharpen_enabled
        return self.sharpen_enabled

    def toggle_grayscale(self):
        self.grayscale_enabled = not self.grayscale_enabled
        return self.grayscale_enabled

    def toggle_invert(self):
        self.invert_enabled = not self.invert_enabled
        return self.invert_enabled

    def toggle_mirror(self):
        self.mirror_enabled = not self.mirror_enabled
        return self.mirror_enabled

# 이 클래스를 사용하기 위한 예시 함수
def process_video(video_path):
    yolo_processor = YoloProcessor()
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = yolo_processor.process_frame(frame)
        cv2.imshow('Processed Frame', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            yolo_processor.toggle_pause()
        elif key == ord('m'):
            yolo_processor.toggle_mosaic()
        # 다른 키 입력에 대한 처리 추가 가능

    cap.release()
    cv2.destroyAllWindows()

# 이 파일을 직접 실행할 때의 테스트 코드
if __name__ == "__main__":
    video_path = 0  # 웹캠 사용, 파일 경로를 지정하면 비디오 파일 사용
    process_video(video_path)