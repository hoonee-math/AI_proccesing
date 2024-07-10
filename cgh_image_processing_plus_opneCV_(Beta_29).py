'''
>> Beta 29
동영상 정지 및 일시정지 버튼 기능 개선
HSV 경계선 검출 오류 수정, HSV 경계선 검출, 가우시안 검출 함수 삭제
원본 이미지와 처리된 이미지 나란히 표시
'''

from tkinter import *                       #(1-3)
from tkinter import ttk     # Beta_3 추가
from tkinter import messagebox
from tkinter.filedialog import *
from tkinter.simpledialog import *
import math
import cv2      #(3-3)
import numpy as np
from tkinter import messagebox, filedialog     # Beta_3 추가
import os.path #파일 처리 / 다음부터는 복붙
from PIL import Image, ImageTk  # Beta_5 필요한 lib import # Beta 20 초기화면 인미지 추가
import threading                # Beta_5 필요한 lib import
import time                     # Beta_5 필요한 lib import
from ultralytics import YOLO    # Beta 6 Yolo lib import
import cv2                      # Beta 6 Yolo lib import
from tkinter import font as tkfont
import logging
from tkinter import Entry, StringVar, Label     # Beta 28 에서 경고를 효과를 위해 추가

## 함수 선언부 ##
def create_button(canvas, x, y, text, command, bg_image):
    # 배경 이미지에서 버튼 영역 추출
    button_width = 150
    button_height = 50
    bg_button = bg_image.crop((x, y, x + button_width, y + button_height))
    bg_button_tk = ImageTk.PhotoImage(bg_button)

    # 버튼 생성
    button_id = canvas.create_image(x, y, image=bg_button_tk, anchor=NW)
    # 그림자 텍스트 생성 (약간 오프셋)
    shadow_id = canvas.create_text(x + button_width / 2 + 5, y + button_height / 2 + 5, text=text,
                                   fill="#808080", font=("Eraser", 135, "bold"))
    text_id = canvas.create_text(x + button_width / 2, y + button_height / 2, text=text,
                                 fill="#FFFFFF", font=("Eraser", 135, "bold"))

    # 마우스 이벤트 처리
    def on_enter(event):
        canvas.itemconfig(text_id, fill="#A75D47")
        canvas.itemconfig(shadow_id, fill="#C2906D")


    def on_leave(event):
        canvas.itemconfig(text_id, fill="#FFFFFF")
        canvas.itemconfig(shadow_id, fill="#808080")


    def on_click(event):
        command()

    canvas.tag_bind(button_id, "<Enter>", on_enter)
    canvas.tag_bind(button_id, "<Leave>", on_leave)
    canvas.tag_bind(button_id, "<Button-1>", on_click)
    canvas.tag_bind(text_id, "<Enter>", on_enter)
    canvas.tag_bind(text_id, "<Leave>", on_leave)
    canvas.tag_bind(text_id, "<Button-1>", on_click)

    return button_id, text_id, bg_button_tk  # 참조 유지를 위해 이미지 객체 반환
def create_initial_screen():
    global main_frame, menu_frame, canvas_frame, initial_frame, canvas  # Beta 24 canvas 변수 추가

    reset_all_variables()  # Beta 24 모든 변수 초기화

    # 기존 프레임 숨기기
    main_frame.pack_forget()

    # Beta 24 캔버스가 존재하면 삭제
    if canvas:
        canvas.destroy()
        canvas = None

    # 초기 화면 프레임 생성
    initial_frame = Frame(window)
    initial_frame.pack(fill=BOTH, expand=True)

    # 배경 이미지 로드
    bg_image = Image.open("initial_page_2.png")
    bg_image = bg_image.resize((1600, 900), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)

    # 배경 이미지를 표시할 캔버스 생성
    bg_canvas = Canvas(initial_frame, width=1600, height=900)
    bg_canvas.pack(fill=BOTH, expand=True)
    bg_canvas.create_image(0, 0, image=bg_photo, anchor="nw")
    bg_canvas.image = bg_photo  # 참조 유지

    # 버튼 생성
    image_button = create_button(bg_canvas, 350, 150, "Photo", switch_to_image_processing, bg_image)
    video_button = create_button(bg_canvas, 1150, 150, "Video", switch_to_video_processing, bg_image)
def malloc2D(h, w, initValue=0) :  #(6-5) 3차원배열을 미리 준비하자, default 파라미터를 사용하면 아주 편해요~
    memory = [[initValue for _ in range(w)] for _ in range(h)]
    return memory   # C 언어 계열을 제외하고는 메모리 해제를 해줄 필요가없음
def malloc3D(h, w, t, initValue=0) :  #(3-3) 3차원배열을 미리 준비하자, default 파라미터를 사용하면 아주 편해요~
    memory = [[[initValue for _ in range(w)] for _ in range(h)] for _ in range(t)]
    return memory   # C 언어 계열을 제외하고는 메모리 해제를 해줄 필요가없음
def OnCV2OutImage():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    outH = outPhoto.shape[0]
    outW = outPhoto.shape[1]
    # 메모리 할당
    outImage = malloc3D(outH, outW, RGB)
    # openCV 개체 --> 입력 이미지
    for i in range(outH):
        for k in range(outW):
            if (len(outPhoto.shape)==3) :  # 칼라 이미지인 경우
                outImage[GG][i][k] = outPhoto.item(i, k, GG)
                outImage[BB][i][k] = outPhoto.item(i, k, RR)
                outImage[RR][i][k] = outPhoto.item(i, k, BB)
            else :                         # 그레이스케일 이미지인 경우
                r = g = b = outPhoto.item(i,k)
                outImage[GG][i][k] = r
                outImage[BB][i][k] = g
                outImage[RR][i][k] = b
def OnSaveDocument() :  #(5-4)
    global window, canvas, paper, inImage, outImage     #(4-1) global로 전역변수 설정해주자!
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    saveFp = asksaveasfile(parent = window, mode = 'wb', defaultextension='*.png',    #cpp write binary
        filetypes = (("Color 이미지", "*.png"), ("All files", "*.*")))
    #비어있는 openCV 개체 생성
    savePhoto = np.zeros((outH,outW, RGB), np.uint8)
    for i in range(outH):
        for k in range(outW):
            tup = tuple( (outImage[BB][i][k], outImage[GG][i][k], outImage[RR][i][k]))
            savePhoto[i,k] = tup
    cv2.imwrite(saveFp.name, savePhoto)
    messagebox.showinfo('성공', saveFp.name + '으로 저장됨')
def OnCloseDocument() :  #(5-5)
    global window, canvas, paper, inImage, outImage     #(4-1) global로 전역변수 설정해주자!
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
def OnOpenDocument():
    global filename, inPhoto, inImage, outImage, inH, inW, outH, outW, video_capture, canvas, video_path

    reset_video_capture()  # 기존 비디오 캡처 객체 해제
    reset_yolo_state()  # Beta 24 YOLO 상태 초기화
    reset_all_variables()  # beta 28 에서 새로 생성된 변수를 위해 추가


    filename = filedialog.askopenfilename(
        filetypes=(("이미지/비디오 파일", "*.jpg;*.png;*.bmp;*.tif;*.mp4;*.avi;*.mov;*.mkv"),
                   ("모든 파일", "*.*")))
    print(filename)
    if filename == '' or filename is None:
        return

    video_path = filename  # Beta 14 선택된 파일의 경로를 video_path에 저장

    file_extension = filename.split('.')[-1].lower()

    if file_extension in ['jpg', 'png', 'bmp', 'tif']:
        inPhoto = cv2.imread(filename)
        inH, inW = inPhoto.shape[:2]
        inImage = malloc3D(inH, inW, RGB)

        # OpenCV image to inImage
        for i in range(inH):
            for k in range(inW):
                inImage[RR][i][k] = inPhoto.item(i, k, 2)
                inImage[GG][i][k] = inPhoto.item(i, k, 1)
                inImage[BB][i][k] = inPhoto.item(i, k, 0)

        equalImage()  # This will create outImage and call OnDraw()
        create_image_menu()
        pass
    elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
        try:
            video_capture = cv2.VideoCapture(filename)
            if not video_capture.isOpened():
                messagebox.showerror("Error", "비디오 파일을 열 수 없습니다.")
                return

            # 캔버스 프레임의 현재 너비 가져오기
            canvas_width = canvas_frame.winfo_width()

            # 첫 프레임 읽기 및 크기 조정
            ret, frame = video_capture.read()
            if ret:
                frame = resize_frame(frame, canvas_width)
                canvas_height = frame.shape[0]
            else:
                messagebox.showerror("Error", "비디오 프레임을 읽을 수 없습니다.")
                return

            # 비디오용 새 캔버스 생성
            if canvas:
                canvas.destroy()
            canvas = Canvas(canvas_frame, width=canvas_width, height=canvas_height)
            canvas.pack()

            create_video_menu()
            update_yolo_button_text()  # beta 24 YOLO 버튼 텍스트 업데이트
            play_video()  # Beta 24 자동으로 비디오 재생 시작
        except Exception as e:
            messagebox.showerror("Error", f"비디오 파일 열기 중 오류 발생: {str(e)}")
            return
    else:
        messagebox.showerror("Error", "지원하지 않는 파일 형식입니다.")
        return

    setup_ui()  # UI 재설정
    sbar.configure(text=filename.split('/')[-1])
def OnDraw():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename, canvas_frame

    # 기존 위젯들 제거
    for widget in canvas_frame.winfo_children():
        widget.destroy()

    # 캔버스 생성
    canvas = Canvas(canvas_frame)
    canvas.pack(side=LEFT, expand=True, fill=BOTH)

    # 캔버스 프레임의 현재 크기 가져오기
    canvas_width = canvas_frame.winfo_width()
    canvas_height = canvas_frame.winfo_height()

    # 이미지 크기 조정
    aspect_ratio = inH / inW
    if canvas_height / (canvas_width / 2) < aspect_ratio:
        new_height = canvas_height
        new_width = int(canvas_height / aspect_ratio)
    else:
        new_width = canvas_width // 2
        new_height = int((canvas_width / 2) * aspect_ratio)

    # 원본 이미지 준비
    in_image = np.zeros((inH, inW, 3), dtype=np.uint8)
    for i in range(inH):
        for k in range(inW):
            in_image[i, k] = [inImage[BB][i][k], inImage[GG][i][k], inImage[RR][i][k]]

    resized_in_image = cv2.resize(in_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    in_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_in_image, cv2.COLOR_BGR2RGB)))

    # 처리된 이미지 준비
    out_image = np.zeros((outH, outW, 3), dtype=np.uint8)
    for i in range(outH):
        for k in range(outW):
            out_image[i, k] = [outImage[BB][i][k], outImage[GG][i][k], outImage[RR][i][k]]

    resized_out_image = cv2.resize(out_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    out_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_out_image, cv2.COLOR_BGR2RGB)))

    # 이미지 표시
    canvas.create_image(0, (canvas_height - new_height) // 2, image=in_photo, anchor=NW)
    canvas.create_image(canvas_width // 2, (canvas_height - new_height) // 2, image=out_photo, anchor=NW)
    canvas.in_photo = in_photo  # 참조 유지
    canvas.out_photo = out_photo  # 참조 유지

    canvas.config(scrollregion=canvas.bbox(ALL))
def setup_ui():  # Beta 4 수정사항, steup_ui() 함수 추가
    global filename, menu_frame

    # 기존 메뉴 항목 제거
    for widget in menu_frame.winfo_children():
        widget.destroy()

    if filename is None or filename == '':
        create_initial_menu()
    else:
        file_extension = filename.split('.')[-1].lower()
        if file_extension in ['jpg', 'png', 'bmp', 'tif']:
            create_image_menu()
        elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
            create_video_menu()
        else:
            create_initial_menu()
def add_home_button():  # Beta 24 함수 추가
    global menu_frame
    Button(menu_frame, text="홈 화면", command=go_to_home).pack(fill=X, padx=10, pady=5)
def go_to_home():       # Beta 24 함수 추가
    create_initial_screen()
    setup_ui()  # UI를 초기 상태로 재설정
def reset_all_variables():  # Beta 24 모든 전역 변수를 초기화하는 함수 추가
    global inImage, outImage, inH, inW, outH, outW, inPhoto, outPhoto, filename
    global video_capture, video_playing, video_path, model, paused
    global mosaic_enabled, blur_enabled, sharpen_enabled, grayscale_enabled
    global invert_enabled, mirror_enabled, hsv_emboss_enabled
    global video_thread, yolo_enabled, object_counts, tracked_ids
    global object_threshold, count_threshold, warning_active, warning_toggle

    inImage, outImage = [], []
    inH, inW, outH, outW = [0] * 4
    inPhoto, outPhoto = None, None
    filename = None

    if video_capture is not None:
        video_capture.release()
    video_capture = None
    video_playing = False
    video_path = None

    # YOLO 관련 변수 초기화
    yolo_enabled = False
    object_counts = {}
    tracked_ids = set()
    if 'model' in globals():
        del model  # YOLO 모델 객체 제거
    model = None

    paused = False

    mosaic_enabled = False
    blur_enabled = False
    sharpen_enabled = False
    grayscale_enabled = False
    invert_enabled = False
    mirror_enabled = False
    hsv_emboss_enabled = False

    if video_thread is not None:
        video_thread.join()
    video_thread = None

    # 새로 추가된 변수 초기화
    if 'object_threshold' in globals() and object_threshold is not None:
        object_threshold.set('')
    else:
        object_threshold = None

    if 'count_threshold' in globals() and count_threshold is not None:
        count_threshold.set('')
    else:
        count_threshold = None

    warning_active = False
    warning_toggle = False

    # YOLO 버튼 텍스트 업데이트
    update_yolo_button_text()

### 비디오 재생 함수 ###
def play_video():
    global video_capture, video_playing, canvas, video_thread
    print("play_video 함수 시작")  # 디버그 출력
    try:
        if video_capture is None or not video_capture.isOpened():
            print("비디오 캡처 객체 오류")  # 디버그 출력
            messagebox.showerror("Error", "비디오 파일을 열 수 없습니다.")
            return

        print(f"비디오 파일 경로: {video_path}")  # 디버그 출력
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        video_playing = True
        print("비디오 재생 스레드 시작")  # 디버그 출력
        video_thread = threading.Thread(target=process_video, daemon=True)
        video_thread.start()
    except Exception as e:
        print(f"오류 발생: {e}")  # 디버그 출력
        messagebox.showerror("Error", f"비디오 재생 중 오류 발생: {e}")
def video_play_thread():
    global video_playing
    video_playing = True
    process_video()
def stop_video():
    global video_capture, video_playing, video_thread
    video_playing = False
    if video_thread is not None:
        video_thread.join()  # 스레드가 완전히 종료될 때까지 대기
    if video_capture is not None:
        video_capture.release()
        video_capture = None  # 비디오 캡처 객체 초기화
    canvas.delete("all")
def pause_video():
    global video_playing
    video_playing = not video_playing  # 일시정지 상태를 토글
def resize_frame(frame, target_width):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    target_height = int(target_width / aspect_ratio)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
def reset_video_capture():
    global video_capture, video_playing
    if video_capture is not None:
        video_playing = False
        video_capture.release()
        video_capture = None
def use_camera():   # Beta 24 비디오 재생 중 영상으로 넘어올때 발생하는 문제 해결
    global video_capture, video_playing, canvas, video_path, canvas_frame

    reset_video_capture()  # beta 24 비디오 초기화
    reset_yolo_state()  # beta 24 YOLO 상태 초기화
    reset_all_variables()

    # Beta 24 기존 비디오 캡처 객체 해제 함수 수정
    if video_capture is not None:
        video_playing = False
        video_capture.release()
    # Beta 24 기존 캔버스 삭제
    if canvas:
        canvas.destroy()

    # 카메라 사용
    video_path = 0
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        messagebox.showerror("에러", "카메라를 열 수 없습니다.")
        return

    # 캔버스 프레임의 현재 너비 가져오기
    canvas_width = canvas_frame.winfo_width()

    # 첫 프레임 읽기 및 크기 조정
    ret, frame = video_capture.read()
    if ret:
        frame = resize_frame(frame, canvas_width)
        canvas_height = frame.shape[0]
    else:
        messagebox.showerror("에러", "카메라 프레임을 읽을 수 없습니다.")
        return

    # 새 캔버스 생성
    canvas = Canvas(canvas_frame, width=canvas_width, height=canvas_height)
    canvas.pack(fill=BOTH, expand=True)

    video_playing = True
    process_video()

    create_video_menu()
    update_yolo_button_text()  # YOLO 버튼 텍스트 업데이트
def start_camera():  # Beta 6 캠 사용 버튼 추가, 초기 메뉴에서 바로 비디오 메뉴 버튼과 카메라 사용을 동시에 사용할수있는 버튼 추가
    create_video_menu()
    use_camera()
def process_video():
    global video_capture, video_playing, canvas, model, yolo_enabled, canvas_frame, object_counts, tracked_ids, sbar
    global object_threshold, count_threshold, warning_active, warning_toggle

    if not video_playing:
        window.after(30, process_video)
        return

    ret, frame = video_capture.read()
    if not ret:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        window.after(30, process_video)
        return

    canvas_width = canvas_frame.winfo_width()
    frame = resize_frame(frame, canvas_width)

    if yolo_enabled and model is not None:
        try:
            results = model.track(frame, persist=True)
            frame_ = results[0].plot()

            current_frame_counts = {}
            for box in results[0].boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                if class_name not in current_frame_counts:
                    current_frame_counts[class_name] = 0
                current_frame_counts[class_name] += 1

            for box in results[0].boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]

                if hasattr(box, 'id') and box.id is not None:
                    obj_id = int(box.id[0])
                    if obj_id not in tracked_ids:
                        tracked_ids.add(obj_id)
                        if class_name not in object_counts:
                            object_counts[class_name] = 0
                        object_counts[class_name] += 1

            update_object_count_display(current_frame_counts)

            # 객체 수 확인 및 경고 처리
            if object_threshold is not None and count_threshold is not None:
                check_object_count(current_frame_counts)
        except Exception as e:
            print(f"YOLO 처리 중 오류 발생: {e}")
            frame_ = frame
    else:
        frame_ = frame
        if 'object_count_label' in globals() and object_count_label:
            object_count_label.config(text="YOLO 비활성화")

    # 효과 적용 (기존 코드와 동일)
    if hsv_emboss_enabled:
        frame_ = apply_hsv_emboss(frame_)
    if mosaic_enabled and yolo_enabled:
        frame_ = apply_mosaic_to_persons(frame_, results)
    if blur_enabled:
        frame_ = apply_blur(frame_)
    if sharpen_enabled:
        frame_ = apply_sharpen(frame_)
    if grayscale_enabled:
        frame_ = apply_grayscale(frame_)
    if invert_enabled:
        frame_ = apply_invert(frame_)
    if mirror_enabled:
        frame_ = apply_mirror(frame_)

    # 경고 효과 적용 (깜빡이는 효과)
    if warning_active:
        if warning_toggle:
            frame_ = cv2.addWeighted(frame_, 1, np.full(frame_.shape, (0, 0, 255), dtype=np.uint8), 0.3, 0)
        warning_toggle = not warning_toggle

    # 상태 표시
    frame_ = add_status_text(frame_)

    # 화면에 표시
    frame_rgb = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))

    if not hasattr(canvas, 'image_on_canvas'):
        canvas.image_on_canvas = canvas.create_image(0, 0, image=photo, anchor=NW)
    else:
        canvas.itemconfig(canvas.image_on_canvas, image=photo)

    canvas.image = photo
    canvas.config(width=frame_.shape[1], height=frame_.shape[0])

    # 상태바 업데이트
    status_text = "비디오 재생 중 - 효과: "
    effects = []
    if yolo_enabled: effects.append("YOLO")
    if mosaic_enabled: effects.append("Mosaic")
    if blur_enabled: effects.append("Blur")
    if sharpen_enabled: effects.append("Sharpen")
    if grayscale_enabled: effects.append("Grayscale")
    if invert_enabled: effects.append("Invert")
    if mirror_enabled: effects.append("Mirror")
    if hsv_emboss_enabled: effects.append("HSV Emboss")
    status_text += ", ".join(effects) if effects else "없음"
    sbar.config(text=status_text)

    window.update_idletasks()
    window.after(30, process_video)
# def update_object_count_display(): # Beta 26 에서 수정하면서 삭제
#     global object_counts, object_count_label
#     if 'object_count_label' not in globals() or not object_count_label:
#         object_count_label = Label(menu_frame, text="", justify=LEFT)
#         object_count_label.pack(side=BOTTOM, fill=X)
#
#     count_text = "탐지된 객체:\n"
#     for obj, count in object_counts.items():
#         count_text += f"{obj}: {count}\n"
#
#     # 레이블이 여전히 존재하는지 확인
#     try:
#         object_count_label.config(text=count_text)
#     except TclError:
#         print("객체 카운트 레이블이 존재하지 않습니다.")
#         # 레이블 재생성
#         object_count_label = Label(menu_frame, text=count_text, justify=LEFT)
#         object_count_label.pack(side=BOTTOM, fill=X)
# def update_object_count_display():    # 현재 화면 객체수 추가 파악을 위해 beta 26 에서 다시 삭제
#     global object_counts, canvas
#
#     if not canvas:
#         return
#
#     # 기존 객체 카운트 텍스트 삭제
#     canvas.delete("object_count")
#
#     count_text = "누적 탐지된 객체:\n"
#     for obj, count in object_counts.items():
#         count_text += f"{obj}: {count}\n"
#
#     if not object_counts:
#         count_text += "없음"
#
#     # 캔버스 크기 가져오기
#     canvas_width = canvas.winfo_width()
#     canvas_height = canvas.winfo_height()
#
#     # 텍스트 위치 및 스타일 설정
#     x = 10  # 왼쪽 여백
#     y = canvas_height - 10  # 아래쪽 여백
#     font = ('Arial', 20, 'bold')  # 폰트 크기를 크게 설정
#     fill_color = 'white'  # 텍스트 색상
#     shadow_color = 'black'  # 그림자 색상
#
#     # 그림자 효과를 위해 텍스트를 약간 오프셋하여 그리기
#     canvas.create_text(x+2, y+2, text=count_text, anchor='sw', font=font, fill=shadow_color, tags="object_count")
#     canvas.create_text(x, y, text=count_text, anchor='sw', font=font, fill=fill_color, tags="object_count")
#
#     # 캔버스 업데이트
#     canvas.update()
def update_object_count_display(current_frame_counts):
    global object_counts, canvas

    if not canvas:
        return

    # 기존 객체 카운트 텍스트 삭제
    canvas.delete("object_count")

    total_count_text = "탐지된 객체:\n"
    current_count_text = "현재 화면:\n"

    for obj in set(list(object_counts.keys()) + list(current_frame_counts.keys())):
        total_count = object_counts.get(obj, 0)
        current_count = current_frame_counts.get(obj, 0)
        total_count_text += f"{obj}: {total_count}\n"
        current_count_text += f"{obj}: {current_count}\n"

    if not object_counts:
        total_count_text += "없음"
    if not current_frame_counts:
        current_count_text += "없음"

    # 캔버스 크기 가져오기
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    # 텍스트 위치 및 스타일 설정
    x_total = 10  # 왼쪽 여백 (기존 위치)
    x_current = x_total + 200  # 현재 프레임 카운트를 위한 위치
    y = canvas_height - 10  # 아래쪽 여백
    font = ('Arial', 20, 'bold')  # 폰트 크기
    fill_color = 'white'
    shadow_color = 'black'

    # 총 탐지된 객체 텍스트 (기존 위치)
    canvas.create_text(x_total+2, y+2, text=total_count_text, anchor='sw', font=font, fill=shadow_color, tags="object_count")
    canvas.create_text(x_total, y, text=total_count_text, anchor='sw', font=font, fill=fill_color, tags="object_count")

    # 현재 화면 객체 텍스트 (새로운 위치)
    canvas.create_text(x_current+2, y+2, text=current_count_text, anchor='sw', font=font, fill=shadow_color, tags="object_count")
    canvas.create_text(x_current, y, text=current_count_text, anchor='sw', font=font, fill=fill_color, tags="object_count")

    # 캔버스 업데이트
    canvas.update()


## 메뉴 관련 함수 ##  Beta_3 create_image_menu 로 모든 메뉴 옮겨오기
def switch_to_image_processing():
    global initial_frame, main_frame
    initial_frame.pack_forget()
    main_frame.pack(fill=BOTH, expand=True)
    setup_ui()
    create_image_menu()
def switch_to_video_processing():
    global initial_frame, main_frame
    initial_frame.pack_forget()
    main_frame.pack(fill=BOTH, expand=True)
    setup_ui()
    create_video_menu()
def create_image_menu():
    global menu_frame, selected_main_menu, selected_sub_menu, sbar

    selected_main_menu = None
    selected_sub_menu = None

    for widget in menu_frame.winfo_children():
        widget.destroy()

    add_home_button()  # Beta 24 홈 화면 버튼 추가

    default_font = ('TkDefaultFont', 8)  # beta 27 메뉴 눌렀다가 다른 메뉴 눌렀을 때 크기가 돌아오지 않는 문제 해결 기본 폰트 설정

    Button(menu_frame, text="파일 열기", command=OnOpenDocument).pack(fill=X, padx=10, pady=5)

    # 구분선 추가
    ttk.Separator(menu_frame, orient='horizontal').pack(fill=X, padx=5, pady=10)

    # 상위 메뉴 버튼들
    pixel_btn = Button(menu_frame, text="화소 처리 기법", command=lambda: toggle_submenu(pixel_frame, pixel_btn), font=default_font)
    pixel_btn.pack(fill=X, padx=10, pady=5)

    geometric_btn = Button(menu_frame, text="기하학 처리", command=lambda: toggle_submenu(geometric_frame, geometric_btn), font=default_font)
    geometric_btn.pack(fill=X, padx=10, pady=5)

    area_btn = Button(menu_frame, text="화소 영역 처리", command=lambda: toggle_submenu(area_frame, area_btn), font=default_font)
    area_btn.pack(fill=X, padx=10, pady=5)

    opencv_btn = Button(menu_frame, text="Open CV 처리", command=lambda: toggle_submenu(opencv_frame, opencv_btn), font=default_font)
    opencv_btn.pack(fill=X, padx=10, pady=5)

    # 서브메뉴 프레임들
    pixel_frame = Frame(menu_frame)
    geometric_frame = Frame(menu_frame)
    area_frame = Frame(menu_frame)
    opencv_frame = Frame(menu_frame)

    # 화소 처리 기법 서브메뉴
    var = StringVar(value="")  # 라디오 버튼 변수
    Button(pixel_frame, text="동일 이미지", command=lambda: (var.set("동일"), select_submenu(equalImage, pixel_frame, "동일 이미지")), font=default_font).pack(fill=X)
    Button(pixel_frame, text="밝게/어둡게", command=lambda: (var.set("밝게/어둡게"), select_submenu(addImage, pixel_frame, "밝게/어둡게")), font=default_font).pack(fill=X)
    Button(pixel_frame, text="그레이스케일", command=lambda: (var.set("그레이"), select_submenu(grayImage, pixel_frame, "그레이스케일")), font=default_font).pack(fill=X)
    Button(pixel_frame, text="반전", command=lambda: (var.set("반전"), select_submenu(reversedImage, pixel_frame, "반전")), font=default_font).pack(fill=X)
    Button(pixel_frame, text='흑백', command=lambda: (var.set("흑백"), select_submenu(bwImage, pixel_frame, "흑백")), font=default_font).pack(fill=X)
    Button(pixel_frame, text='포스터라이징', command=lambda: (var.set("포스터"), select_submenu(posterImage, pixel_frame, "포스터라이징")), font=default_font).pack(fill=X)

    # 기하학 처리 서브메뉴
    Button(geometric_frame, text="확대/축소", command=lambda: select_submenu(zoomImage, geometric_frame, "확대/축소"), font=default_font).pack(fill=X)
    Button(geometric_frame, text="이동", command=lambda: select_submenu(moveImage, geometric_frame, "이동"), font=default_font).pack(fill=X)
    Button(geometric_frame, text='대칭', command=lambda: select_submenu(mirrorImage, geometric_frame, "대칭"), font=default_font).pack(fill=X)
    Button(geometric_frame, text='회전', command=lambda: select_submenu(rotateImage, geometric_frame, "회전"), font=default_font).pack(fill=X)

    # 화소 영역 처리 서브메뉴
    Button(area_frame, text="엠보싱(RGB)", command=lambda: select_submenu(embossImageRGB, area_frame, "엠보싱(RGB)"), font=default_font).pack(fill=X)
    Button(area_frame, text='엠보싱(HSV)', command=lambda: select_submenu(embossImageHSV, area_frame, "엠보싱(HSV)"), font=default_font).pack(fill=X)
    Button(area_frame, text='블러링(RGB)', command=lambda: select_submenu(blurImageRGB, area_frame, "블러링(RGB)"), font=default_font).pack(fill=X)
    Button(area_frame, text='블러링(HSV)', command=lambda: select_submenu(blurImageHSV, area_frame, "블러링(HSV)"), font=default_font).pack(fill=X)
    Button(area_frame, text='샤프닝(RGB)', command=lambda: select_submenu(sharpRGB, area_frame, "샤프닝(RGB)"), font=default_font).pack(fill=X)
    Button(area_frame, text='샤프닝(HSV)', command=lambda: select_submenu(sharpHSV, area_frame, "샤프닝(HSV)"), font=default_font).pack(fill=X)
    Button(area_frame, text='경계선(RGB)', command=lambda: select_submenu(lineRGB, area_frame, "경계선(RGB)"), font=default_font).pack(fill=X)
    # Button(area_frame, text='경계선(HSV)', command=lambda: select_submenu(lineHSV, area_frame, "경계선(HSV)"), font=default_font).pack(fill=X)
    Button(area_frame, text='가우시안(RGB)', command=lambda: select_submenu(gaussianRGB, area_frame, "가우시안(RGB)"), font=default_font).pack(fill=X)
    # Button(area_frame, text='가우시안(HSV)', command=lambda: select_submenu(gaussianHSV, area_frame, "가우시안(HSV)"), font=default_font).pack(fill=X)

    # Open CV 처리 서브메뉴
    Button(opencv_frame, text='동일 이미지', command=lambda: select_submenu(equalImageCV, opencv_frame, "동일 이미지"), font=default_font).pack(fill=X)
    Button(opencv_frame, text='그레이 이미지', command=lambda: select_submenu(grayscaleImageCV, opencv_frame, "그레이 이미지"), font=default_font).pack(fill=X)
    Button(opencv_frame, text='HSV 변환', command=lambda: select_submenu(hsvImageCV, opencv_frame, "HSV 변환"), font=default_font).pack(fill=X)
    Button(opencv_frame, text='binary 이미지', command=lambda: select_submenu(binaryImageCV, opencv_frame, "binary 이미지"), font=default_font).pack(fill=X)
    Button(opencv_frame, text='적응형 이미지', command=lambda: select_submenu(binary2ImageCV, opencv_frame, "적응형 이미지"), font=default_font).pack(fill=X)
    Button(opencv_frame, text='엠보싱 이미지', command=lambda: select_submenu(embossImageCV, opencv_frame, "엠보싱 이미지"), font=default_font).pack(fill=X)
    Button(opencv_frame, text='카툰 이미지', command=lambda: select_submenu(cartonImageCV, opencv_frame, "카툰 이미지"), font=default_font).pack(fill=X)
    Button(opencv_frame, text='얼굴인식', command=lambda: select_submenu(faceDetectCV, opencv_frame, "얼굴인식"), font=default_font).pack(fill=X)
    Button(opencv_frame, text='코인식', command=lambda: select_submenu(noseDetectCV, opencv_frame, "코인식"), font=default_font).pack(fill=X)

    # 구분선 추가
    ttk.Separator(menu_frame, orient='horizontal').pack(fill=X, padx=5, pady=10)
def toggle_submenu(frame, button):  # beta 27에서 defalut 추가를 위해 전면 수정
    global selected_main_menu
    default_font = ('TkDefaultFont', 8)
    selected_font = ('TkDefaultFont', 10, 'bold')

    if selected_main_menu:
        selected_main_menu.config(font=default_font)
    selected_main_menu = button
    button.config(font=selected_font)

    for widget in menu_frame.winfo_children():
        if isinstance(widget, Frame):
            widget.pack_forget()

    frame.pack(fill=X, padx=10, pady=5)
    update_status_bar()  # 상태바 업데이트 함수 호출 # beta 26 에서 추가
def update_status_bar():    # beta 26 에서 추가
    global sbar, selected_main_menu, selected_sub_menu
    if selected_main_menu and selected_sub_menu:
        sbar.config(text=f"상태바: {selected_main_menu.cget('text')} - {selected_sub_menu.cget('text')}")
    elif selected_main_menu:
        sbar.config(text=f"상태바: {selected_main_menu.cget('text')}")
    else:
        sbar.config(text="상태바:")
def select_submenu(command, frame, name):
    global selected_sub_menu, sbar, selected_main_menu
    default_font = ('TkDefaultFont', 8)    # beta 27 에서 제대로 초기화하기 위해 추가
    selected_font = ('TkDefaultFont', 10, 'bold')

    for btn in frame.winfo_children():
        if isinstance(btn, Button):
            btn.config(font=default_font)   # beta 27 에서 제대로 초기화하기 위해 추가

    for btn in frame.winfo_children():
        if isinstance(btn, Button) and btn.cget("text") == name:
            selected_sub_menu = btn
            btn.config(font=selected_font)  # beta 27 에서 제대로 초기화하기 위해 추가
            sbar.config(text=f"상태바: {selected_main_menu.cget('text')} - {name}") # beta 26 에서 추가
            break
    command()
def create_video_menu():
    global menu_frame, video_path, person_count_label, yolo_button, object_count_label, selected_video_button
    global object_threshold, count_threshold
    for widget in menu_frame.winfo_children():
        widget.destroy()

    selected_video_button = None  # 초기화

    add_home_button()  # Beta 24 홈 화면 버튼 추가
    Button(menu_frame, text="파일 열기", command=OnOpenDocument).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="카메라 사용", command=use_camera).pack(fill=X, padx=10, pady=5)

    # 구분선 추가
    ttk.Separator(menu_frame, orient='horizontal').pack(fill=X, padx=5, pady=10)

    # 재생 제어 섹션
    Button(menu_frame, text="재생", command=play_video).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="일시정지", command=pause_video).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="정지", command=stop_video).pack(fill=X, padx=10, pady=5)

    # 구분선 추가
    ttk.Separator(menu_frame, orient='horizontal').pack(fill=X, padx=5, pady=10)

    # 이미지 효과 섹션
    effects_frame = Frame(menu_frame)
    effects_frame.pack(fill=X, padx=5, pady=5)

    Label(effects_frame, text="영상 처리 효과", font=("Helvetica", 10, "bold")).pack(fill=X, pady=5)

    effect_buttons = [
        ("블러", toggle_blur),
        ("선명화", toggle_sharpen),
        ("그레이스케일", toggle_grayscale),
        ("반전", toggle_invert),
        ("미러", toggle_mirror),
        ("HSV 엠보싱", toggle_hsv_emboss)
    ]

    for text, command in effect_buttons:
        btn = Button(effects_frame, text=text)
        btn.config(command=lambda b=btn, c=command: toggle_button_state(b, c))
        btn.pack(fill=X, padx=5, pady=2)

    # 구분선 추가
    ttk.Separator(menu_frame, orient='horizontal').pack(fill=X, padx=5, pady=10)

    # YOLO 및 카메라 섹션
    yolo_button = Button(menu_frame, text="YOLO 적용" if not yolo_enabled else "YOLO 비활성화")
    yolo_button.config(command=lambda: toggle_button_state(yolo_button, toggle_yolo))
    yolo_button.pack(fill=X, padx=10, pady=5)

    # 모자이크 버튼 추가
    mosaic_button = Button(menu_frame, text="모자이크")
    mosaic_button.config(command=lambda: toggle_button_state(mosaic_button, toggle_mosaic))
    mosaic_button.pack(fill=X, padx=10, pady=5)

    # 구분선 추가
    ttk.Separator(menu_frame, orient='horizontal').pack(fill=X, padx=5, pady=10)

    # 객체 이름 입력 필드
    object_threshold = StringVar()
    Label(menu_frame, text="객체 이름:").pack(fill=X, padx=5, pady=2)
    Entry(menu_frame, textvariable=object_threshold).pack(fill=X, padx=5, pady=2)

    # 객체 수 임계값 입력 필드
    count_threshold = StringVar()
    Label(menu_frame, text="객체 수 임계값:").pack(fill=X, padx=5, pady=2)
    Entry(menu_frame, textvariable=count_threshold).pack(fill=X, padx=5, pady=2)

    # 실시간 모니터링 시작
    window.after(500, monitor_thresholds)

    # 객체 카운트 표시 레이블
    object_count_label = Label(menu_frame, text="YOLO 비활성화", justify=LEFT)
    object_count_label.pack(side=BOTTOM, fill=X)
def monitor_thresholds():   # Beta 28 에서 경고를 효과를 위해 추가, creat_video_menu()에서 사용
    global object_threshold, count_threshold

    # 입력값 확인
    object_name = object_threshold.get()
    count_value = count_threshold.get()

    if not object_name:
        show_warning("객체 이름을 입력하세요")
    elif not count_value:
        show_warning("객체 수 임계값을 입력하세요")
    elif not count_value.isdigit() or int(count_value) <= 10:
        show_warning("올바른 객체 수 임계값을 입력하세요 (10 초과)")
    else:
        hide_warning()

    # 0.5초 후 다시 확인
    window.after(500, monitor_thresholds)
def show_warning(message):  # Beta 28 에서 경고를 효과를 위해 함수 추가
    global warning_label
    if 'warning_label' not in globals() or not warning_label:
        warning_label = Label(menu_frame, text=message, fg="red")
        warning_label.pack(fill=X, padx=5, pady=2)
    else:
        warning_label.config(text=message)
def hide_warning(): # Beta 28 에서 경고를 효과를 위해 함수 추가
    global warning_label
    if 'warning_label' in globals() and warning_label:
        warning_label.pack_forget()
        warning_label = None
def check_object_count(current_frame_counts):   # Beta 28 에서 경고를 효과를 위해 함수 추가
    global object_threshold, count_threshold, warning_active

    object_name = object_threshold.get()
    count_value = count_threshold.get()

    if object_name and count_value and count_value.isdigit():
        threshold = int(count_value)    # Beta 28 threshold 이 입력받은 값
        if object_name in current_frame_counts and current_frame_counts[object_name] >= threshold:
            if not warning_active:
                warning_active = True
                flash_warning()
        else:
            warning_active = False
def flash_warning():    # Beta 28 에서 경고를 효과를 위해 함수 추가
    global warning_active, canvas
    if warning_active:
        canvas.config(bg='red')
        window.after(500, lambda: canvas.config(bg='white'))
        window.after(1000, flash_warning)

def toggle_button_state(button, command):   # beta 27 에서 동영상 메뉴 bole 처리를 위해 추가
    global selected_video_button
    if selected_video_button:
        selected_video_button.config(font=('TkDefaultFont', 10))
    button.config(font=('TkDefaultFont', 10, 'bold'))
    selected_video_button = button
    command()
def create_initial_menu():
    global menu_frame
    Button(menu_frame, text="파일 열기", command=OnOpenDocument).pack(fill=X, padx=10, pady=5)  # 첫 화면의 왼쪽 파일 열기 버튼 padx, pady : 여백
    Button(menu_frame, text="카메라 사용", command=start_camera).pack(fill=X, padx=10, pady=5)    # Beta 6 캠 사용 버튼 추가
    # 다른 초기 버튼들을 여기에 추가할 수 있습니다.

### 영상 처리 함수 ###
def equalImage() :
    global window, canvas, paper, inImage, outImage     # global로 전역변수 설정해주자!
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    # 중요! 출력 이미지의 크기를 결정 ---> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc3D(outH, outW, RGB)
    ### 진짜 영상처리 알고리즘 ###
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                outImage[rgb][i][k] = inImage[rgb][i][k]
    #########################
    OnDraw()
def addImage() : #(5-3)
    global window, canvas, paper, inImage, outImage, sbar     # beta 27 에서 sbar 추가
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    # (4-10) 중요! 출력 이미지의 크기를 결정 ---> 알고리즘에 의존
    outH = inH
    outW = inW
    # (4-11) 메모리 할당
    outImage = malloc3D(outH, outW, RGB)
    ### (4-12) 진짜 영상처리 알고리즘 ###
    value = askinteger('밝게/어둡게', '밝게/어둡게 값', minvalue=-255, maxvalue=255)
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                if(inImage[rgb][i][k] + value < 0) :
                    outImage[rgb][i][k] = 0
                elif(inImage[rgb][i][k] + value > 255) :
                    outImage[rgb][i][k] = 255
                else:
                    outImage[rgb][i][k] = inImage[rgb][i][k] +value
    # 상태바 업데이트
    sbar.config(text=f"상태바: 화소 처리 기법 - 밝게/어둡게 (값: {value})")

    #########################
    OnDraw() # (4-13)
def grayImage():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### grayscale 알고리즘 ###
    for i in range(inH):
        for k in range(inW):
            outImage[RR][i][k] = int((inImage[RR][i][k] + inImage[GG][i][k] + inImage[BB][i][k]) / 3)
            outImage[GG][i][k] = int((inImage[RR][i][k] + inImage[GG][i][k] + inImage[BB][i][k]) / 3)
            outImage[BB][i][k] = int((inImage[RR][i][k] + inImage[GG][i][k] + inImage[BB][i][k]) / 3)
    #########################
    OnDraw() # (4-13)
def reversedImage():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### grayscale 알고리즘 ###
    for i in range(inH):
        for k in range(inW):
            outImage[RR][i][k] = 255-inImage[RR][i][k]
            outImage[GG][i][k] = 255-inImage[GG][i][k]
            outImage[BB][i][k] = 255-inImage[BB][i][k]
    #########################
    OnDraw()
def embossImageRGB():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### embossing 알고리즘 ###
    MSIZE = 3
    mask = [    [  -1,  0,  0],
                [   0,  0,  0],
                [   0,  0,  1]  ]
    # 임시 출력, 출력 영상 확보
    tmpInImage = malloc3D(inH+2, inW+2, RGB)
    tmpOutImage = malloc3D(inH, inW, RGB)
    # 원본 -> 임시
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i+1][k+1] = inImage[rgb][i][k]
    # 회선 연산
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                s = 0.0
                for m in range(MSIZE):
                    for n in range(MSIZE):
                        s += mask[m][n] * tmpInImage[rgb][i+m][k+n]
                    tmpOutImage[rgb][i][k]=s
    # 후 처리(127더하기)
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                tmpOutImage[rgb][i][k] += 127
    # 임시 출력 -> 실제 출력
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if(tmpOutImage[rgb][i][k]<0):
                    outImage[rgb][i][k]=0
                elif(tmpOutImage[rgb][i][k]>255):
                    outImage[rgb][i][k]=255
                else :
                    outImage[rgb][i][k]= int(tmpOutImage[rgb][i][k])
    #########################
    OnDraw()
import  colorsys #RGB를 HSV로 변환
def embossImageHSV(): #(6-2)
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    # (6-3) 입력 RGB 메모리 -> 입력 HSV 메모리
    inImageHSV = malloc3D(inH, inW, RGB)
    # (6-4) RGB 칼라 모델 -> HSV 칼라 모델 변경 // c에서 직접 코드를 구현하려면 정말 복잠. 여기서는 colorsys 를 import해서 편하게 사용가능
    for i in range(inH):
        for k in range(inW):
            r, g, b = inImage[RR][i][k], inImage[GG][i][k], inImage[BB][i][k]
            h, s, v = colorsys.rgb_to_hsv(r,g,b)
            inImageHSV[0][i][k], inImageHSV[1][i][k], inImageHSV[2][i][k] = h, s, v
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### embossing 알고리즘 ###
    MSIZE = 3
    mask = [    [  -1,  0,  0],
                [   0,  0,  0],
                [   0,  0,  1]  ]
    # 임시 출력, 출력 영상 확보 (6-6)   # 엠보싱을 위해 2D 메모리 할당을 하나 더 만들자 (6-4)
    tmpInImageV = malloc2D(inH+2, inW+2)
    tmpOutImageV = malloc2D(inH, inW)
    # 원본 -> 임시
    for i in range(inH):
        for k in range(inW):
            tmpInImageV[i+1][k+1] = inImageHSV[2][i][k]
    # 회선 연산
    for i in range(inH):
        for k in range(inW):
            s = 0.0
            for m in range(MSIZE):
                for n in range(MSIZE):
                    s += mask[m][n] * tmpInImageV[i+m][k+n]
                tmpOutImageV[i][k] = s
    # 후 처리(127더하기)
    for i in range(outH):
        for k in range(outW):
            if(tmpOutImageV[i][k] + 127 < 0) :
                tmpOutImageV[i][k] = 0
            elif (tmpOutImageV[i][k] + 127 > 255) :
                tmpOutImageV[i][k] = 255
            else :
                tmpOutImageV[i][k] += 127
    # 임시 출력을 RGB로 변환해서 실제로 출력하기
    for i in range(outH):
        for k in range(outW):
            h, s, v = inImageHSV[0][i][k], inImageHSV[1][i][k], tmpOutImageV[i][k]
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            outImage[RR][i][k], outImage[GG][i][k], outImage[BB][i][k] = int(r), int(g), int(b),
    #########################
    OnDraw()
def bwImage():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### grayscale 알고리즘 ###
    for i in range(inH):
        for k in range(inW):
            z = int((inImage[RR][i][k] + inImage[GG][i][k] + inImage[BB][i][k]) / 3)
            if z<128:
                outImage[RR][i][k] = 0
                outImage[GG][i][k] = 0
                outImage[BB][i][k] = 0
            else:
                outImage[RR][i][k] = 255
                outImage[GG][i][k] = 255
                outImage[BB][i][k] = 255


    #########################
    OnDraw() # (4-13)
def posterImage():
    global window, canvas, paper, inImage, outImage, sbar     # beta 27 에서 sbar 추가
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### grayscale 알고리즘 ###
    level = askinteger('포스터라이징', '포스터라이징 단계',minvalue=1,maxvalue=100)
    for i in range(inH):
        for k in range(inW):
            for z in range(level):
                lower_bound = z * 255 / level
                upper_bound = (z + 1) * 255 / level
                if lower_bound <= inImage[RR][i][k] < upper_bound:
                    outImage[RR][i][k] = int(z * 255 / level)
                if lower_bound <= inImage[GG][i][k] < upper_bound:
                    outImage[GG][i][k] = int(z * 255 / level)
                if lower_bound <= inImage[BB][i][k] < upper_bound:
                    outImage[BB][i][k] = int(z * 255 / level)
    # 상태바 업데이트
    sbar.config(text=f"상태바: 화소 처리 기법 - 포스터라이징 (단계: {level})")
    #########################
    OnDraw()
def mirrorImage():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### grayscale 알고리즘 ###
    for i in range(inH):
        for k in range(inW):
            outImage[RR][i][k] = inImage[RR][i][inW-k-1]
            outImage[GG][i][k] = inImage[GG][i][inW-k-1]
            outImage[BB][i][k] = inImage[BB][i][inW-k-1]
    #########################
    OnDraw()
def zoomImage():
    global window, canvas, paper, inImage, outImage, sbar     # beta 27 에서 sbar 추가
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    scale = askfloat('이미지 크기 조절', '비율(0~3)',minvalue=0.01, maxvalue=3.1)
    outH = int(inH * scale)  # 출력 이미지의 크기를 결정
    outW = int(inW * scale)
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### zoom 알고리즘 ###
    for i in range(outH):
        for k in range(outW):
            i_s = int(i/scale)
            k_s = int(k/scale)
            outImage[RR][i][k] = inImage[RR][i_s][k_s]
            outImage[GG][i][k] = inImage[GG][i_s][k_s]
            outImage[BB][i][k] = inImage[BB][i_s][k_s]
    # 상태바 업데이트
    sbar.config(text=f"상태바: 기하학 처리 - 배율 (단계: {scale})")
    #########################
    OnDraw()
class MyDialog(Dialog):
    def body(self, master):
        Label(master, text="x:").grid(row=0)
        Label(master, text="y:").grid(row=1)

        self.e1 = Entry(master)
        self.e2 = Entry(master)

        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        return self.e1 # initial focus

    def apply(self):
        self.result = (self.e1.get(), self.e2.get())
def moveImage():
    global window, canvas, paper, inImage, outImage, sbar     # beta 27 에서 sbar 추가
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    root = Tk()         #x, y축 방향으로 이동할 거리 입력 받기
    d = MyDialog(root)
    [x_axis, y_axis] = d.result
    x = int(x_axis)
    y = int(y_axis)
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### zoom 알고리즘 ###
    for i in range(outH):
        for k in range(outW):
            if i<x or k<y :
                outImage[RR][i][k] = 255
                outImage[GG][i][k] = 255
                outImage[BB][i][k] = 255
            else:
                outImage[RR][i][k] = inImage[RR][i-x][k-y]
                outImage[GG][i][k] = inImage[GG][i-x][k-y]
                outImage[BB][i][k] = inImage[BB][i-x][k-y]
    # 상태바 업데이트
    sbar.config(text=f"상태바: 기하학 처리 - 이동 (x값 변화량: {x}, y값 변화량: {y})")
    #########################
    OnDraw()

def rotationImage():
    global window, canvas, paper, inImage, outImage, sbar     # beta 27 에서 sbar 추가
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    CenterH = int(inH / 2)
    CenterW = int(inW / 2)
    PI = 3.14159265358979
    angle = askinteger('회전', '회전 각도',minvalue=1,maxvalue=100)
    Radian = angle * PI / 180.0

    for i in range(outH):
        for k in range(outW):
            newH = int((i - CenterH) * math.cos(Radian) + (k - CenterW) * math.sin(Radian) + CenterH)
            newW = int(-(i - CenterH) * math.sin(Radian) + (k - CenterW) * math.cos(Radian) + CenterW)

            if (newH >= 0 and newH < inH and newW >= 0 and newW < inW):
                outImage[RR][i][k] = inImage[RR][newH][newW]
                outImage[GG][i][k] = inImage[GG][newH][newW]
                outImage[BB][i][k] = inImage[BB][newH][newW]
            else:
                outImage[RR][i][k] = 0  # 이동 후 빈 영역 0으로 처리
                outImage[GG][i][k] = 0
                outImage[BB][i][k] = 0

    # 상태바 업데이트
    sbar.config(text=f"상태바: 기하학 처리 (회전 각도: {angle})")
def rotateImage():
    global window, canvas, paper, inImage, outImage, sbar     # beta 27 에서 sbar 추가
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    angle = askinteger('회전', '회전 각도', minvalue=1, maxvalue=360)
    radian = angle * math.pi / 180.0  # 각도를 라디안으로 변환
    cos_val = math.cos(radian)
    sin_val = math.sin(radian)
    x_center = inW // 2  # 중심점 x 좌표
    y_center = inH // 2  # 중심점 y 좌표

    # 출력 이미지 메모리 확보
    outH = inH
    outW = inW
    outImage = malloc3D(outH, outW, RGB)

    # 회전 변환
    for i in range(inH):
        for k in range(inW):
            new_i = int((i - y_center) * cos_val - (k - x_center) * sin_val + y_center)
            new_k = int((i - y_center) * sin_val + (k - x_center) * cos_val + x_center)
            if 0 <= new_i < outH and 0 <= new_k < outW:
                outImage[RR][new_i][new_k] = inImage[RR][i][k]
                outImage[GG][new_i][new_k] = inImage[GG][i][k]
                outImage[BB][new_i][new_k] = inImage[BB][i][k]

    # 상태바 업데이트
    sbar.config(text=f"상태바: 기하학 처리 (회전 각도: {angle})")
    # 결과 이미지를 화면에 출력
    OnDraw()
def blurImageHSV():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    # (6-3) 입력 RGB 메모리 -> 입력 HSV 메모리
    inImageHSV = malloc3D(inH, inW, RGB)
    # (6-4) RGB 칼라 모델 -> HSV 칼라 모델 변경 // c에서 직접 코드를 구현하려면 정말 복잠. 여기서는 colorsys 를 import해서 편하게 사용가능
    for i in range(inH):
        for k in range(inW):
            r, g, b = inImage[RR][i][k], inImage[GG][i][k], inImage[BB][i][k]
            h, s, v = colorsys.rgb_to_hsv(r,g,b)
            inImageHSV[0][i][k], inImageHSV[1][i][k], inImageHSV[2][i][k] = h, s, v
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### embossing 알고리즘 ###
    MSIZE = 3
    mask = [    [  -2, -1,  0],
                [  -1,  1,  1],
                [   0,  1,  2]  ]
    # 임시 출력, 출력 영상 확보 (6-6)   # 엠보싱을 위해 2D 메모리 할당을 하나 더 만들자 (6-4)
    tmpInImageV = malloc2D(inH+2, inW+2)
    tmpOutImageV = malloc2D(inH, inW)
    # 원본 -> 임시
    for i in range(inH):
        for k in range(inW):
            tmpInImageV[i+1][k+1] = inImageHSV[2][i][k]
    # 회선 연산
    for i in range(inH):
        for k in range(inW):
            s = 0.0
            for m in range(MSIZE):
                for n in range(MSIZE):
                    s += mask[m][n] * tmpInImageV[i+m][k+n]
                tmpOutImageV[i][k] = s
    # 후 처리(127더하기)
    for i in range(outH):
        for k in range(outW):
            if(tmpOutImageV[i][k] + 127 < 0) :
                tmpOutImageV[i][k] = 0
            elif (tmpOutImageV[i][k] + 127 > 255) :
                tmpOutImageV[i][k] = 255
            else :
                tmpOutImageV[i][k] += 127
    # 임시 출력을 RGB로 변환해서 실제로 출력하기
    for i in range(outH):
        for k in range(outW):
            h, s, v = inImageHSV[0][i][k], inImageHSV[1][i][k], tmpOutImageV[i][k]
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            outImage[RR][i][k], outImage[GG][i][k], outImage[BB][i][k] = int(r), int(g), int(b),
    #########################
    OnDraw()
def blurImageRGB():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### embossing 알고리즘 ###
    MSIZE = 3
    mask = [    [  -2, -1,  0],
                [  -1,  1,  1],
                [   0,  1,  2]  ]
    # 임시 출력, 출력 영상 확보
    tmpInImage = malloc3D(inH+2, inW+2, RGB)
    tmpOutImage = malloc3D(inH, inW, RGB)
    # 원본 -> 임시
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i+1][k+1] = inImage[rgb][i][k]
    # 회선 연산
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                s = 0.0
                for m in range(MSIZE):
                    for n in range(MSIZE):
                        s += mask[m][n] * tmpInImage[rgb][i+m][k+n]
                    tmpOutImage[rgb][i][k]=s
    # 후 처리(127더하기)
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                tmpOutImage[rgb][i][k] += 127
    # 임시 출력 -> 실제 출력
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if(tmpOutImage[rgb][i][k]<0):
                    outImage[rgb][i][k]=0
                elif(tmpOutImage[rgb][i][k]>255):
                    outImage[rgb][i][k]=255
                else :
                    outImage[rgb][i][k]= int(tmpOutImage[rgb][i][k])
    #########################
    OnDraw()
def sharpRGB():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### embossing 알고리즘 ###
    MSIZE = 3
    mask = [    [  -1, -1, -1],
                [  -1,  9, -1],
                [  -1, -1, -1]  ]
    # 임시 출력, 출력 영상 확보
    tmpInImage = malloc3D(inH+2, inW+2, RGB)
    tmpOutImage = malloc3D(inH, inW, RGB)
    # 원본 -> 임시
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i+1][k+1] = inImage[rgb][i][k]
    # 회선 연산
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                s = 0.0
                for m in range(MSIZE):
                    for n in range(MSIZE):
                        s += mask[m][n] * tmpInImage[rgb][i+m][k+n]
                    tmpOutImage[rgb][i][k]=s
    # 후 처리(127더하기)
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                tmpOutImage[rgb][i][k] += 127
    # 임시 출력 -> 실제 출력
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if(tmpOutImage[rgb][i][k]<0):
                    outImage[rgb][i][k]=0
                elif(tmpOutImage[rgb][i][k]>255):
                    outImage[rgb][i][k]=255
                else :
                    outImage[rgb][i][k]= int(tmpOutImage[rgb][i][k])
    #########################
    OnDraw()
def sharpHSV():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    # (6-3) 입력 RGB 메모리 -> 입력 HSV 메모리
    inImageHSV = malloc3D(inH, inW, RGB)
    # (6-4) RGB 칼라 모델 -> HSV 칼라 모델 변경 // c에서 직접 코드를 구현하려면 정말 복잠. 여기서는 colorsys 를 import해서 편하게 사용가능
    for i in range(inH):
        for k in range(inW):
            r, g, b = inImage[RR][i][k], inImage[GG][i][k], inImage[BB][i][k]
            h, s, v = colorsys.rgb_to_hsv(r,g,b)
            inImageHSV[0][i][k], inImageHSV[1][i][k], inImageHSV[2][i][k] = h, s, v
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### embossing 알고리즘 ###
    MSIZE = 3
    mask = [    [  -1, -1, -1],
                [  -1,  9, -1],
                [  -1, -1, -1]  ]
    # 임시 출력, 출력 영상 확보 (6-6)   # 엠보싱을 위해 2D 메모리 할당을 하나 더 만들자 (6-4)
    tmpInImageV = malloc2D(inH+2, inW+2)
    tmpOutImageV = malloc2D(inH, inW)
    # 원본 -> 임시
    for i in range(inH):
        for k in range(inW):
            tmpInImageV[i+1][k+1] = inImageHSV[2][i][k]
    # 회선 연산
    for i in range(inH):
        for k in range(inW):
            s = 0.0
            for m in range(MSIZE):
                for n in range(MSIZE):
                    s += mask[m][n] * tmpInImageV[i+m][k+n]
                tmpOutImageV[i][k] = s
    # 후 처리(127더하기)
    for i in range(outH):
        for k in range(outW):
            if(tmpOutImageV[i][k] + 127 < 0) :
                tmpOutImageV[i][k] = 0
            elif (tmpOutImageV[i][k] + 127 > 255) :
                tmpOutImageV[i][k] = 255
            else :
                tmpOutImageV[i][k] += 127
    # 임시 출력을 RGB로 변환해서 실제로 출력하기
    for i in range(outH):
        for k in range(outW):
            h, s, v = inImageHSV[0][i][k], inImageHSV[1][i][k], tmpOutImageV[i][k]
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            outImage[RR][i][k], outImage[GG][i][k], outImage[BB][i][k] = int(r), int(g), int(b),
    #########################
    OnDraw()

def maskRGB(a,b1,c,d,e,f,g1,h,i):
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### embossing 알고리즘 ###
    MSIZE = 3
    mask = [    [  a, b1, c],
                [  d, e, f],
                [  g1, h, i]  ]
    # 임시 출력, 출력 영상 확보
    tmpInImage = malloc3D(inH+2, inW+2, RGB)
    tmpOutImage = malloc3D(inH, inW, RGB)
    # 원본 -> 임시
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i+1][k+1] = inImage[rgb][i][k]
    # 회선 연산
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                s = 0.0
                for m in range(MSIZE):
                    for n in range(MSIZE):
                        s += int(mask[m][n] * tmpInImage[rgb][i+m][k+n])
                    tmpOutImage[rgb][i][k]=s
    # 후 처리(127더하기)
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                tmpOutImage[rgb][i][k] += 127
    # 임시 출력 -> 실제 출력
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if(tmpOutImage[rgb][i][k]<0):
                    outImage[rgb][i][k]=0
                elif(tmpOutImage[rgb][i][k]>255):
                    outImage[rgb][i][k]=255
                else :
                    outImage[rgb][i][k]= int(tmpOutImage[rgb][i][k])
    #########################
    OnDraw()
def maskHSV(a,b1,c,d,e,f,g1,h,i):
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    # (6-3) 입력 RGB 메모리 -> 입력 HSV 메모리
    inImageHSV = malloc3D(inH, inW, RGB)
    # (6-4) RGB 칼라 모델 -> HSV 칼라 모델 변경 // c에서 직접 코드를 구현하려면 정말 복잠. 여기서는 colorsys 를 import해서 편하게 사용가능
    for i in range(inH):
        for k in range(inW):
            r, g, b = inImage[RR][i][k], inImage[GG][i][k], inImage[BB][i][k]
            h, s, v = colorsys.rgb_to_hsv(r,g,b)
            inImageHSV[0][i][k], inImageHSV[1][i][k], inImageHSV[2][i][k] = h, s, v
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### embossing 알고리즘 ###
    MSIZE = 3
    mask = [    [   a,  b1,  c],
                [   d,  e,  f],
                [   g1,  h,  i]  ]
    # 임시 출력, 출력 영상 확보 (6-6)   # 엠보싱을 위해 2D 메모리 할당을 하나 더 만들자 (6-4)
    tmpInImageV = malloc2D(inH+2, inW+2)
    tmpOutImageV = malloc2D(inH, inW)
    # 원본 -> 임시
    for i in range(inH):
        for k in range(inW):
            tmpInImageV[i+1][k+1] = inImageHSV[2][i][k]
    # 회선 연산
    for i in range(inH):
        for k in range(inW):
            s = 0.0
            for m in range(MSIZE):
                for n in range(MSIZE):
                    s += int(mask[m][n] * tmpInImageV[i+m][k+n])
                tmpOutImageV[i][k] = s
    ## 후 처리(127더하기)
    #for i in range(outH):
    #    for k in range(outW):
    #        if(tmpOutImageV[i][k] + 127 < 0) :
    #            tmpOutImageV[i][k] = 0
   #         elif (tmpOutImageV[i][k] + 127 > 255) :
   #             tmpOutImageV[i][k] = 255
    #        else :
    #            tmpOutImageV[i][k] += 127
    # 임시 출력을 RGB로 변환해서 실제로 출력하기
    for i in range(outH):
        for k in range(outW):
            h, s, v = inImageHSV[0][i][k], inImageHSV[1][i][k], tmpOutImageV[i][k]
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            outImage[RR][i][k], outImage[GG][i][k], outImage[BB][i][k] = int(r), int(g), int(b),
    #########################
    OnDraw()
def maskHSV127(a,b1,c,d,e,f,g1,h,i):
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    # (6-3) 입력 RGB 메모리 -> 입력 HSV 메모리
    inImageHSV = malloc3D(inH, inW, RGB)
    # (6-4) RGB 칼라 모델 -> HSV 칼라 모델 변경 // c에서 직접 코드를 구현하려면 정말 복잠. 여기서는 colorsys 를 import해서 편하게 사용가능
    for i in range(inH):
        for k in range(inW):
            r, g, b = inImage[RR][i][k], inImage[GG][i][k], inImage[BB][i][k]
            h, s, v = colorsys.rgb_to_hsv(r,g,b)
            inImageHSV[0][i][k], inImageHSV[1][i][k], inImageHSV[2][i][k] = h, s, v
    outH = inH  # 출력 이미지의 크기를 결정
    outW = inW
    outImage = malloc3D(outH, outW, RGB) # 메모리 할당
    ### embossing 알고리즘 ###
    MSIZE = 3
    mask = [    [   a,  b1,  c],
                [   d,  e,  f],
                [   g1,  h,  i]  ]
    # 임시 출력, 출력 영상 확보 (6-6)   # 엠보싱을 위해 2D 메모리 할당을 하나 더 만들자 (6-4)
    tmpInImageV = malloc2D(inH+2, inW+2)
    tmpOutImageV = malloc2D(inH, inW)
    # 원본 -> 임시
    for i in range(inH):
        for k in range(inW):
            tmpInImageV[i+1][k+1] = inImageHSV[2][i][k]
    # 회선 연산
    for i in range(inH):
        for k in range(inW):
            s = 0.0
            for m in range(MSIZE):
                for n in range(MSIZE):
                    s += int(mask[m][n] * tmpInImageV[i+m][k+n])
                tmpOutImageV[i][k] = s
    # 후 처리(127더하기)
    for i in range(outH):
        for k in range(outW):
            if(tmpOutImageV[i][k] + 127 < 0) :
                tmpOutImageV[i][k] = 0
            elif (tmpOutImageV[i][k] + 127 > 255) :
                tmpOutImageV[i][k] = 255
            else :
                tmpOutImageV[i][k] += 127
    # 임시 출력을 RGB로 변환해서 실제로 출력하기
    for i in range(outH):
        for k in range(outW):
            h, s, v = inImageHSV[0][i][k], inImageHSV[1][i][k], tmpOutImageV[i][k]
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            outImage[RR][i][k], outImage[GG][i][k], outImage[BB][i][k] = int(r), int(g), int(b),
    #########################
    OnDraw()
def lineRGB():
    rc = askinteger('세로선 가로선 검출', '세로선:1 가로선:0',minvalue=0,maxvalue=1)
    if rc == 1:
        maskRGB(
		-1, 0 ,  1,
		-2, 0 ,  2,
		-1, 0 ,  1)
    else:
        maskRGB(
		-1, -2, -1,
	     0, 0,  0,
		 1,  2,  1)
def lineHSV():
    rc = askinteger('세로선 가로선 검출', '세로선:1 가로선:0',minvalue=0,maxvalue=1)
    if rc == 1:
        maskHSV(
		-1, 0 ,  1,
		-2, 0 ,  2,
		-1, 0 ,  1)
    else:
        maskHSV(
		-1, -2, -1,
	     0, 0,  0,
		 1,  2,  1)
def gaussianRGB():
    maskRGB(
    1/ 16, 1/ 8, 1/ 16,
    1/ 8, 1/ 4, 1/ 8,
    1/ 16, 1/ 8, 1/ 16)
def gaussianHSV():
    maskHSV(
    1/ 16, 1/ 8, 1/ 16,
    1/ 8, 1/ 4, 1/ 8,
    1/ 16, 1/ 8, 1/ 16)

### Open CV 함수부 ###
def equalImageCV():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    ##############
    ## open cv 제공 함수 활용
    outPhoto = inPhoto.copy()
    ##############

    OnCV2OutImage()
    OnDraw()
def grayscaleImageCV():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    ##############
    ## open cv 제공 함수 활용
    outPhoto = inPhoto.copy()
    outPhoto = cv2.cvtColor(inPhoto,cv2.COLOR_BGR2GRAY)
    ##############

    OnCV2OutImage()
    OnDraw()
def hsvImageCV():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    ##############
    ## open cv 제공 함수 활용
    outPhoto = inPhoto.copy()
    outPhoto = cv2.cvtColor(outPhoto,cv2.COLOR_BGR2HSV)
    ##############
    OnCV2OutImage()
    OnDraw()
def binaryImageCV():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    ##############
    ## open cv 제공 함수 활용
    outPhoto = inPhoto.copy()
    _, outPhoto = cv2.threshold(outPhoto, 127, 255, cv2.THRESH_BINARY)
    ##############
    OnCV2OutImage()
    OnDraw()
def binary2ImageCV():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    ##############
    ## open cv 제공 함수 활용
    outPhoto = inPhoto.copy()
    outPhoto = cv2.cvtColor(outPhoto,cv2.COLOR_BGR2GRAY)
    outPhoto = cv2.adaptiveThreshold(outPhoto, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33,-5)
    ##############
    OnCV2OutImage()
    OnDraw()
def embossImageCV():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    ##############
    ## open cv 제공 함수 활용
    outPhoto = inPhoto.copy()
    mask = np.zeros((3,3), np.float32)
    mask[0][0] = -1.0
    mask[2][2] = 1.0
    outPhoto = cv2.filter2D(outPhoto, -1, mask)
    outPhoto += 127
    ##############
    OnCV2OutImage()
    OnDraw()
def cartonImageCV():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    ##############
    ## open cv 제공 함수 활용
    outPhoto = inPhoto.copy()
    outPhoto = cv2.cvtColor(outPhoto, cv2.COLOR_BGR2GRAY)
    outPhoto = cv2.medianBlur(outPhoto,7)
    edges = cv2.Laplacian(outPhoto,cv2.CV_8U,ksize=5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    outPhoto = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
    ##############

    OnCV2OutImage()
    OnDraw()
def faceDetectCV():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    ##############
    ## open cv 제공 함수 활용
    outPhoto = inPhoto.copy()
    ## 그레이 스케일로 전환
    gray = cv2.cvtColor(inPhoto,cv2.COLOR_BGR2GRAY)
    ## 훈련된 모델 불러오기
    clf = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")     #(4-3)훈련된 모델을 줄테니까
    ## 얼굴 찾기(여러개 찾기 s)
    face_rects = clf.detectMultiScale(gray, 1.02, 5) # (4-2)찾아줘! 네모를 찾아줘, 여러개 위치 좌표를 찾아줌
            # ## 화면에 찾은 네모 그리기 (삭제 후 if문 추가)
            # for (x, y, w, h) in face_rects :
            #     cv2.rectangle(outPhoto,(x,y),(x+w, y+h),(0,255,0),0)
    if len(face_rects) == 0:
        # 얼굴을 찾지 못한 경우
        messagebox.showinfo("알림", "얼굴이 있는 이미지가 아닙니다.")
    else:
        # 얼굴을 찾은 경우
        ## 화면에 찾은 네모 그리기
        for (x, y, w, h) in face_rects:
            cv2.rectangle(outPhoto, (x, y), (x + w, y + h), (0, 255, 0), 2)
    ##############
    OnCV2OutImage()
    OnDraw()
def noseDetectCV():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    ##############
    ## open cv 제공 함수 활용
    outPhoto = inPhoto.copy()
    ## 그레이 스케일로 전환
    gray = cv2.cvtColor(inPhoto,cv2.COLOR_BGR2GRAY)
    ## 훈련된 모델 불러오기
    clf = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")     #(4-3)훈련된 모델을 줄테니까
    ## 얼굴 찾기(여러개 찾기 s)
    #sf = askfloat("스케일 팩터", "입력", initialvalue=1.0)
    face_rects = clf.detectMultiScale(gray, 1.03, 2) # (4-2)찾아줘! 네모를 찾아줘, 여러개 위치 좌표를 찾아줌
    ## 화면에 찾은 네모 그리기
    for (x, y, w, h) in face_rects :
        cv2.rectangle(outPhoto,(x,y),(x+w, y+h),(0,255,0),2)
    ##############
    OnCV2OutImage()
    OnDraw()

### yolo 적용 kkh 함수 추가 ###  Beta 8에서 추가
def apply_hsv_emboss(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
    v = cv2.filter2D(v, -1, kernel)
    final_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
def apply_mirror(image):
    return cv2.flip(image, 1)
def apply_mosaic_to_persons(image, results):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if model.names[cls] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_h = int((y2 - y1) / 2)
                image = apply_mosaic(image, x1, y1, x2 - x1, face_h)
    return image
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
def add_status_text(image):
    status_text = []
    if mosaic_enabled: status_text.append("Mosaic: ON")
    if blur_enabled: status_text.append("Blur: ON")
    if sharpen_enabled: status_text.append("Sharpen: ON")
    if grayscale_enabled: status_text.append("Grayscale: ON")
    if invert_enabled: status_text.append("Invert: ON")
    if mirror_enabled: status_text.append("Mirror: ON")
    if hsv_emboss_enabled: status_text.append("HSV emboss: ON")

    if not status_text:
        status_text = ["All Effects Off"]

    y0, dy = 30, 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_color = (255, 255, 255)  # 흰색
    shadow_color = (0, 0, 0)      # 검은색
    shadow_offset = (2, 2)

    for i, line in enumerate(status_text):
        y = y0 + i * dy
        # 그림자 텍스트
        cv2.putText(image, line, (10 + shadow_offset[0], y + shadow_offset[1]),
                    font, font_scale, shadow_color, thickness, cv2.LINE_AA)
        # 메인 텍스트
        cv2.putText(image, line, (10, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return image

### kkh 토글 함수 추가 ###    Beta 8에서 추가
def toggle_mosaic():
    global mosaic_enabled
    mosaic_enabled = not mosaic_enabled
def toggle_blur():
    global blur_enabled
    blur_enabled = not blur_enabled
def toggle_sharpen():
    global sharpen_enabled
    sharpen_enabled = not sharpen_enabled
def toggle_grayscale():
    global grayscale_enabled
    grayscale_enabled = not grayscale_enabled
def toggle_invert():
    global invert_enabled
    invert_enabled = not invert_enabled
def toggle_mirror():
    global mirror_enabled
    mirror_enabled = not mirror_enabled
def toggle_hsv_emboss():
    global hsv_emboss_enabled
    hsv_emboss_enabled = not hsv_emboss_enabled

def toggle_yolo():
    global yolo_enabled, object_count_label, object_counts, tracked_ids, model
    yolo_enabled = not yolo_enabled
    if yolo_enabled:
        if load_yolo_model():
            object_counts = {}
            tracked_ids = set()
            print("YOLO 활성화")
        else:
            yolo_enabled = False
    else:
        object_counts = {}
        tracked_ids = set()
        print("YOLO 비활성화")

    update_yolo_button_text()

    # 객체 카운트 표시 레이블 초기화
    if 'object_count_label' in globals() and object_count_label:
        object_count_label.destroy()
    object_count_label = Label(menu_frame, text="", justify=LEFT)
    object_count_label.pack(side=BOTTOM, fill=X)
    # global yolo_enabled, object_count_label, object_counts, tracked_ids
    # yolo_enabled = not yolo_enabled
    # if not yolo_enabled:
    #     if 'object_count_label' in globals() and object_count_label:
    #         object_count_label.config(text="YOLO 비활성화")
    # else:
    #     object_counts = {}  # YOLO를 활성화할 때 카운트 초기화
    #     tracked_ids = set()  # 추적된 ID 목록도 초기화
# def toggle_yolo():  # Beta 24 YOLO 모델 로딩을 필요할 때만 하도록 수정 후 오류로 되돌림
#     global yolo_enabled, object_count_label, object_counts, tracked_ids
#     yolo_enabled = not yolo_enabled
#     load_yolo_model()
#     if yolo_enabled:
#         object_counts = {}  # YOLO를 활성화할 때 카운트 초기화
#         tracked_ids = set()  # 추적된 ID 목록도 초기화ㅍ
#     else:
#         if 'object_count_label' in globals() and object_count_label:
#             object_count_label.config(text="YOLO 비활성화")
def load_yolo_model():  # Beta 24 YOLO 모델 로딩 함수, YOLO 모델 로딩을 필요할 때만 하도록 수정
    global model
    if model is None:
        try:
            model = YOLO('yolov8n.pt')
            print("YOLO 모델 로드 완료")
        except Exception as e:
            print(f"YOLO 모델 로드 중 오류 발생: {e}")
            messagebox.showerror("Error", f"YOLO 모델 로드 실패: {e}")
            return False
    return True
        # global model
        # if model is None:
        #     try:
        #         model = YOLO('yolov8n.pt')
        #         logging.info("YOLO 모델 로드 성공")
        #     except Exception as e:
        #         logging.error(f"YOLO 모델 로드 실패: {e}")
        #         messagebox.showerror("오류", f"YOLO 모델을 로드할 수 없습니다: {e}")
    # if 'model' not in globals():
    #     model = YOLO('yolov8n.pt')
def reset_yolo_state():
    global yolo_enabled, object_counts, tracked_ids, model
    yolo_enabled = False
    object_counts = {}
    tracked_ids = set()
    model = None  # YOLO 모델을 초기화합니다.
def update_yolo_button_text():  # beta 24 추가
    global yolo_button
    if 'yolo_button' in globals() and yolo_button:
        yolo_button.config(text="YOLO 비활성화" if yolo_enabled else "YOLO 적용")

## 전역 변수부 ##
window, canvas, paper = None, None, None    #(1-1)
inImage, outImage = [], []      #(3-1) unsigned char **m_inImage.... 영상처리를 위한 전역변수 선언!
inH, inW, outH, outW = [0]*4
inPhoto, outPhoto = None, None      #(3-2) OpenCV를 사용하려면 설치해야함(p38)
filename = None
RGB, RR, GG, BB = 3, 0, 1, 2    #(4-5)
video_capture = None    # Beta 5: 비디오 관련 전역 변수 추가
video_playing = False   # Beta 5: 비디오 관련 전역 변수 추가
global video_path
video_path = None     # Beta 5: 비디오 관련 전역 변수 추가
model = YOLO('yolov8n.pt')  # Beta 8에서 추가
paused = False  # Beta 8에서 추가
mosaic_enabled = False  # Beta 8에서 추가
blur_enabled = False  # Beta 8에서 추가
sharpen_enabled = False  # Beta 8에서 추가
grayscale_enabled = False  # Beta 8에서 추가
invert_enabled = False  # Beta 8에서 추가
mirror_enabled = False  # Beta 8에서 추가
hsv_emboss_enabled = False  # Beta 8에서 추가
video_thread = None  # Beta 11 비디오 재생 스레드를 관리하기 위한 변수
yolo_enabled = False  # YOLO 활성화 상태를 위한 변수 추가
person_count_label = None
object_counts = {}  # Beta 18 전역 변수 추가
tracked_ids = set() # Beta 18
model = None
selected_main_menu, selected_sub_menu = None, None
selected_video_button = None  # beta 27 전역 변수로 선언
object_threshold = None     # Beta 28 에서 경고 효과를 위해 추가
count_threshold = None      # Beta 28 에서 경고 효과를 위해 추가
warning_active = False      # Beta 28 에서 경고 효과를 위해 추가
warning_toggle = False      # Beta 28 에서 경고 효과를 위해 추가

## 메인 코드부 ##
window = Tk()
window.title("Photo & Video Tool")
window.geometry("1600x900")

# 전체 레이아웃을 위한 프레임 (초기에는 숨김)
main_frame = Frame(window)

# 왼쪽 메뉴 프레임 (고정 너비)
menu_frame = Frame(main_frame, width=100, bg='lightgray')
menu_frame.pack(side=LEFT, fill=Y)
menu_frame.pack_propagate(False)  # 프레임 크기 고정

# 오른쪽 캔버스 프레임
canvas_frame = Frame(main_frame)
canvas_frame.pack(side=RIGHT, fill=BOTH, expand=True)

# 상태바
sbar = Label(window, text="상태바", bd=1, relief=SUNKEN, anchor=W)
sbar.pack(side=BOTTOM, fill=X)

# 초기 화면 생성
create_initial_screen()

window.mainloop()