import cv2
import numpy as np
## 함수
def ssdNet(image, confVal) :
    CONF_VALUE = confVal #함수는 안건드리는게 좋으니 confVal를 직접 받아서 쓰자, 인식률 인정값
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
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_VALUE:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return image
## 변수
## 메인
filename = "images(DL)/zoopark.mp4"
#capture = cv2.VideoCapture(filename)
capture = cv2.VideoCapture(0)
s_factor = 1 # 화면 크기 비율 조정 : 원래 크기로 할건지 아니면 조절할건지
confValue = 0.5 # 인식률 인정값 (70% 이상 인정) - 조절 가능
## AI 영상 인식 ##
#30프레임 이미지인경우 굳이 1초에 30번이나 찾을필요있느냐?! no!
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
frameCount = 0
paused = False   # 일시정지 상태를 추적하는 변수

while True:
    if not paused:
        ret, frame = capture.read()
        if not ret:
            break

        frameCount += 10
        if frameCount % 5 == 0:
            frame = cv2.resize(frame, None,
                    fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)
            frame = ssdNet(frame, confValue)


    cv2.imshow('Video', frame)

    key = cv2.waitKey(20)
    if key == 27:  # ESC
        break
    elif key == ord('p'):  # 'p' 키
        paused = not paused     # 일시정지 상태 토글

capture.release()
cv2.destroyWindow()


