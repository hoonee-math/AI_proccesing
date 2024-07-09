'''
>> Beta 18 객체수 파악 함수 추가
object_counts 전역 변수 추가
process_video 함수 수정
update_object_count_display 함수 추가
create_video_menu 수정
toggle_yolo 수정
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
from PIL import Image, ImageTk  # Beta_5 필요한 lib import
import threading                # Beta_5 필요한 lib import
import time                     # Beta_5 필요한 lib import
from ultralytics import YOLO    # Beta 6 Yolo lib import
import cv2                      # Beta 6 Yolo lib import

## 함수 선언부 ##
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

    # 이미지 표시
    paper = PhotoImage(width=outW, height=outH)
    canvas.create_image(0, 0, image=paper, anchor=NW)
    canvas.config(scrollregion=(0, 0, outW, outH))

    rgbString = ""
    for i in range(outH):
        tmpString = ""
        for k in range(outW):
            r, g, b = outImage[RR][i][k], outImage[GG][i][k], outImage[BB][i][k]
            tmpString += "#%02x%02x%02x " % (r, g, b)
        rgbString += "{" + tmpString + "} "
    paper.put(rgbString)

    canvas.pack(side=LEFT, expand=True, fill=BOTH)

## 메뉴 관련 함수 ##  Beta_3 create_image_menu 로 모든 메뉴 옮겨오기
def create_image_menu():
    global menu_frame
    for widget in menu_frame.winfo_children():
        widget.destroy()

    Button(menu_frame, text="파일 열기", command=OnOpenDocument).pack(fill=X, padx=10, pady=5)

    # 화소 처리 기법 메뉴 ## 메뉴 버튼 method https://blog.naver.com/sisosw/221408635889
    pixel_menu = Menubutton(menu_frame, text="화소 처리 기법", relief=RAISED)
    pixel_menu.pack(fill=X, padx=10, pady=5)
    pixel_menu.menu = Menu(pixel_menu, tearoff=0)
    pixel_menu["menu"] = pixel_menu.menu

    pixel_menu.menu.add_radiobutton(label="동일 이미지", command=equalImage)
    pixel_menu.menu.add_radiobutton(label="밝게/어둡게", command=addImage)
    pixel_menu.menu.add_radiobutton(label="그레이스케일", command=grayImage)
    pixel_menu.menu.add_radiobutton(label="반전", command=reversedImage)
    pixel_menu.menu.add_radiobutton(label='흑백', command=bwImage)
    pixel_menu.menu.add_radiobutton(label='포스터라이징', command=posterImage)

    # 기하학 처리 메뉴
    geometric_menu = Menubutton(menu_frame, text="기하학 처리", relief=RAISED)
    geometric_menu.pack(fill=X, padx=10, pady=5)
    geometric_menu.menu = Menu(geometric_menu, tearoff=0)
    geometric_menu["menu"] = geometric_menu.menu

    geometric_menu.menu.add_command(label="확대/축소", command=zoomImage)
    geometric_menu.menu.add_command(label="이동", command=moveImage)
    geometric_menu.menu.add_command(label='대칭', command=mirrorImage)
    geometric_menu.menu.add_command(label='회전', command=rotateImage)

    # 화소 영역 처리 메뉴
    area_menu = Menubutton(menu_frame, text="화소 영역 처리", relief=RAISED)
    area_menu.pack(fill=X, padx=10, pady=5)
    area_menu.menu = Menu(area_menu, tearoff=0)
    area_menu["menu"] = area_menu.menu

    area_menu.menu.add_command(label="엠보싱(RGB)", command=embossImageRGB)
    area_menu.menu.add_command(label='엠보싱(HSV)', command=embossImageHSV)
    area_menu.menu.add_command(label='블러링(RGB)', command=blurImageRGB)
    area_menu.menu.add_command(label='블러링(HSV)', command=blurImageHSV)
    area_menu.menu.add_command(label='샤프닝(RGB)', command=sharpRGB)
    area_menu.menu.add_command(label='샤프닝(HSV)', command=sharpHSV)
    area_menu.menu.add_command(label='경계선(RGB)', command=lineRGB)
    area_menu.menu.add_command(label='경계선(HSV)', command=lineHSV)
    area_menu.menu.add_command(label='가우시안(RGB)', command=gaussianRGB)
    area_menu.menu.add_command(label='가우시안(HSV)', command=gaussianHSV)

    # Open CV 이미지 처리 메뉴 추가 (예: 히스토그램)
    openCVMenu = Menubutton(menu_frame, text="Open CV 처리", relief=RAISED)
    openCVMenu.pack(fill=X, padx=10, pady=5)
    openCVMenu.menu = Menu(openCVMenu, tearoff=0)
    openCVMenu["menu"] = openCVMenu.menu

    openCVMenu.menu.add_command(label='동일 이미지', command=equalImageCV)
    openCVMenu.menu.add_command(label='그레이 이미지', command=grayscaleImageCV)
    openCVMenu.menu.add_command(label='HSV 변환', command=hsvImageCV)
    openCVMenu.menu.add_command(label='binary 이미지', command=binaryImageCV)
    openCVMenu.menu.add_command(label='적응형 이미지', command=binary2ImageCV)
    openCVMenu.menu.add_command(label='엠보싱 이미지', command=embossImageCV)
    openCVMenu.menu.add_command(label='카툰 이미지', command=cartonImageCV)
    openCVMenu.menu.add_separator()
    openCVMenu.menu.add_command(label='얼굴인식', command=faceDetectCV)
    openCVMenu.menu.add_command(label='코인식', command=noseDetectCV)

    # caffe_Menu 이미지 처리 메뉴 추가 (예: 히스토그램)
    caffe_Menu = Menubutton(menu_frame, text="caffe 영상", relief=RAISED)
    caffe_Menu.pack(fill=X, padx=10, pady=5)
    caffe_Menu.menu = Menu(caffe_Menu, tearoff=0)
    caffe_Menu["menu"] = caffe_Menu.menu

    caffe_Menu.menu.add_command(label='동일 이미지')
    caffe_Menu.menu.add_command(label='Detector')

    yolo_Menu = Menubutton(menu_frame, text="Yolo 영상", relief=RAISED)
    yolo_Menu.pack(fill=X, padx=10, pady=5)
    yolo_Menu.menu = Menu(yolo_Menu, tearoff=0)
    yolo_Menu["menu"] = yolo_Menu.menu

    yolo_Menu.menu.add_command(label='동일 이미지')
    yolo_Menu.menu.add_command(label='Detector') # #
def create_video_menu():
    global menu_frame, video_path, person_count_label # Beta 18 변수 추가 person_count_label
    for widget in menu_frame.winfo_children():
        widget.destroy()

    # 파일 및 재생 제어 섹션
    Button(menu_frame, text="파일 열기", command=OnOpenDocument).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="재생", command=play_video).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="일시정지", command=pause_video).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="정지", command=stop_video).pack(fill=X, padx=10, pady=5)

    # 구분선 추가
    ttk.Separator(menu_frame, orient='horizontal').pack(fill=X, padx=5, pady=10)

    # YOLO 및 카메라 섹션
    Button(menu_frame, text="YOLO 적용", command=toggle_yolo).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="카메라 사용", command=use_camera).pack(fill=X, padx=10, pady=5)

    # 구분선 추가
    ttk.Separator(menu_frame, orient='horizontal').pack(fill=X, padx=5, pady=10)

    # 이미지 효과 섹션
    effects_frame = Frame(menu_frame)
    effects_frame.pack(fill=X, padx=5, pady=5)

    Label(effects_frame, text="이미지 효과", font=("Helvetica", 10, "bold")).pack(fill=X, pady=5)

    Button(effects_frame, text="블러", command=toggle_blur).pack(fill=X, padx=5, pady=2)
    Button(effects_frame, text="선명화", command=toggle_sharpen).pack(fill=X, padx=5, pady=2)
    Button(effects_frame, text="그레이스케일", command=toggle_grayscale).pack(fill=X, padx=5, pady=2)
    Button(effects_frame, text="반전", command=toggle_invert).pack(fill=X, padx=5, pady=2)
    Button(effects_frame, text="미러", command=toggle_mirror).pack(fill=X, padx=5, pady=2)
    Button(effects_frame, text="HSV 엠보싱", command=toggle_hsv_emboss).pack(fill=X, padx=5, pady=2)
    Button(effects_frame, text="모자이크", command=toggle_mosaic).pack(fill=X, padx=5, pady=2)
    # 구분선 추가
    ttk.Separator(menu_frame, orient='horizontal').pack(fill=X, padx=5, pady=10)

    # 객체 카운트 표시 레이블
    object_count_label = Label(menu_frame, text="YOLO 비활성화", justify=LEFT)
    object_count_label.pack(side=BOTTOM, fill=X)
def create_initial_menu():
    global menu_frame
    Button(menu_frame, text="파일 열기", command=OnOpenDocument).pack(fill=X, padx=10, pady=5)  # 첫 화면의 왼쪽 파일 열기 버튼 padx, pady : 여백
    Button(menu_frame, text="카메라 사용", command=start_camera).pack(fill=X, padx=10, pady=5)    # Beta 6 캠 사용 버튼 추가
    # 다른 초기 버튼들을 여기에 추가할 수 있습니다.
def start_camera():  # Beta 6 캠 사용 버튼 추가, 초기 메뉴에서 바로 비디오 메뉴 버튼과 카메라 사용을 동시에 사용할수있는 버튼 추가
    create_video_menu()
    use_camera()
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
    global window, canvas, paper, inImage, outImage     #(4-9) global로 전역변수 설정해주자!
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
    global window, canvas, paper, inImage, outImage
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
    global window, canvas, paper, inImage, outImage
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
    global window, canvas, paper, inImage, outImage
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
    #########################
    OnDraw()

def rotationImage():
    global window, canvas, paper, inImage, outImage
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
def rotateImage():
    global window, canvas, paper, inImage, outImage
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
def stop_video():   # Beta 11
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
    video_playing = False
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
def use_camera():
    global video_capture, video_playing, canvas, video_path, canvas_frame

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
# process_video 함수 수정
def process_video():
    global video_capture, video_playing, canvas, model, yolo_enabled, canvas_frame, object_counts, tracked_ids  # Beta 18 object_counts, tracked_ids 변수 추가

    if not video_playing:
        return

    ret, frame = video_capture.read()
    if not ret:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        window.after(30, process_video)
        return

    # 캔버스 프레임의 현재 너비 가져오기
    canvas_width = canvas_frame.winfo_width()

    # 프레임 크기 조정
    frame = resize_frame(frame, canvas_width)

    # YOLO 모델 적용 (조건부)
    if yolo_enabled:
        results = model.track(frame, persist=True)
        frame_ = results[0].plot()

        # 객체 카운트 업데이트
        for box in results[0].boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]

            # tracking_id가 있는 경우에만 처리
            if hasattr(box, 'id') and box.id is not None:
                obj_id = int(box.id[0])
                if obj_id not in tracked_ids:
                    tracked_ids.add(obj_id)
                    if class_name not in object_counts:
                        object_counts[class_name] = 0
                    object_counts[class_name] += 1

        # 누적 객체 카운트 표시
        update_object_count_display()
    else:
        frame_ = frame
        if 'object_count_label' in globals() and object_count_label:
            object_count_label.config(text="YOLO 비활성화")

    # 효과 적용 (기존 코드와 동일)
    # ... (이전 코드 유지)

    # 화면에 표시
    frame_rgb = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))

    if not hasattr(canvas, 'image_on_canvas'):
        canvas.image_on_canvas = canvas.create_image(0, 0, image=photo, anchor=NW)
    else:
        canvas.itemconfig(canvas.image_on_canvas, image=photo)

    canvas.image = photo
    canvas.config(width=frame_.shape[1], height=frame_.shape[0])

    window.update_idletasks()
    window.after(30, process_video)
def update_object_count_display():
    global object_counts, object_count_label
    if 'object_count_label' not in globals() or not object_count_label:
        object_count_label = Label(menu_frame, text="", justify=LEFT)
        object_count_label.pack(side=BOTTOM, fill=X)

    count_text = "탐지된 객체:\n"
    for obj, count in object_counts.items():
        count_text += f"{obj}: {count}\n"
    object_count_label.config(text=count_text)

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

    y0, dy = 30, 40
    for i, line in enumerate(status_text):
        y = y0 + i * dy
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if not status_text:
        instructions = ["All Effects Off"]
        for i, line in enumerate(instructions):
            y = y0 + i * dy
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
# Beta YOLO 토글 함수 추가
def toggle_yolo():
    global yolo_enabled, object_count_label, object_counts, tracked_ids
    yolo_enabled = not yolo_enabled
    if not yolo_enabled:
        if 'object_count_label' in globals() and object_count_label:
            object_count_label.config(text="YOLO 비활성화")
    else:
        object_counts = {}  # YOLO를 활성화할 때 카운트 초기화
        tracked_ids = set()  # 추적된 ID 목록도 초기화
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



## 메인 코드부 ##
window = Tk()   #(1-2)
window.title("Photo & Video Tool")    # 이름 변경
window.geometry("1000x600")  # 기존 800 600에서 확대 적용
# 전체 레이아웃을 위한 프레임
main_frame = Frame(window)
main_frame.pack(fill=BOTH, expand=True)
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

# 초기 UI 설정
setup_ui()   # Beta 4 수정사항, steup_ui() 함수 추가

window.mainloop()   #(1-4)