import cv2
import numpy as np


## 함수
def ssdNet(image, confVal, prev_detections):
    CONF_VALUE = confVal
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    current_detections = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_VALUE:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            current_detections.append((idx, startX, startY, endX, endY, confidence))

    # 이전 탐지 결과와 현재 탐지 결과를 비교
    final_detections = []
    for prev in prev_detections:
        prev_idx, prev_startX, prev_startY, prev_endX, prev_endY, prev_conf, prev_life = prev
        matched = False
        for curr in current_detections:
            curr_idx, curr_startX, curr_startY, curr_endX, curr_endY, curr_conf = curr
            if curr_idx == prev_idx:
                iou = calculate_iou(prev, curr)
                if iou > 0.5:  # IOU 임계값
                    final_detections.append(
                        (curr_idx, curr_startX, curr_startY, curr_endX, curr_endY, curr_conf, 5))  # 수명 갱신
                    matched = True
                    break
        if not matched:
            prev_life -= 1
            if prev_life > 0:
                final_detections.append(
                    (prev_idx, prev_startX, prev_startY, prev_endX, prev_endY, prev_conf, prev_life))

    # 새로 탐지된 객체 추가
    for curr in current_detections:
        if curr not in [d[:6] for d in final_detections]:
            final_detections.append(curr + (5,))  # 초기 수명 5

    # 결과 그리기
    for (idx, startX, startY, endX, endY, confidence, life) in final_detections:
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    return image, final_detections

def add_progress_bar(frame, current_frame, total_frames):
    height, width = frame.shape[:2]
    bar_height = 20
    bar_width = int(width * (current_frame / total_frames))

    cv2.rectangle(frame, (0, height - bar_height), (bar_width, height), (0, 255, 0), -1)
    cv2.rectangle(frame, (0, height - bar_height), (width, height), (255, 255, 255), 2)

    return frame
def on_mouse_click(event, x, y, flags, param):
    global frameCount, total_frames, capture
    if event == cv2.EVENT_LBUTTONDOWN:
        height = param[0]
        if y > height - 20:  # 클릭이 progress bar 영역 내에 있는지 확인
            frameCount = int((x / param[1]) * total_frames)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frameCount)
def calculate_iou(box1, box2):
    # IOU (Intersection over Union) 계산
    x1 = max(box1[1], box2[1])
    y1 = max(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[3] - box1[1]) * (box1[4] - box1[2])
    area2 = (box2[3] - box2[1]) * (box2[4] - box2[2])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


# 나머지 함수들은 그대로 유지...

## 메인
filename = "images(DL)/zoopark.mp4"
capture = cv2.VideoCapture(filename)

s_factor = 1
confValue = 0.5
bar_t = 5

total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
frameCount = 0
paused = False

cv2.namedWindow('Video')
frame = capture.read()[1]
height, width = frame.shape[:2]
cv2.setMouseCallback('Video', on_mouse_click, param=[height, width])

prev_detections = []

while True:
    if not paused:
        ret, frame = capture.read()
        if not ret:
            break

        frameCount += 3
        if frameCount % 5 == 0:
            frame = cv2.resize(frame, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)
            frame, prev_detections = ssdNet(frame, confValue, prev_detections)

    frame = add_progress_bar(frame, frameCount, total_frames)
    cv2.imshow('Video', frame)

    key = cv2.waitKey(20)
    if key == 27:  # ESC
        break
    elif key == ord('p'):  # 'p' 키
        paused = not paused
    elif key == 81:  # 왼쪽 방향키
        move_time(-1, bar_t)
    elif key == 83:  # 오른쪽 방향키
        move_time(1, bar_t)

capture.release()
cv2.destroyAllWindows()