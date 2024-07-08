from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import *
from tkinter.simpledialog import *
import math
import cv2
import numpy as np
import os.path
import _pyinstaller_hooks_contrib
from ultralytics import YOLO
from PIL import Image, ImageTk


### Alpha 2 수정사항###
'''

'''

## 함수 선언부
def malloc2D(h, w, initValue=0) :  #(6-5) 3차원배열을 미리 준비하자, default 파라미터를 사용하면 아주 편해요~
    memory = [[initValue for _ in range(w)] for _ in range(h)]
    return memory   # C 언어 계열을 제외하고는 메모리 해제를 해줄 필요가없음
def malloc3D(h, w, t, initValue=0) :  #(3-3) 3차원배열을 미리 준비하자, default 파라미터를 사용하면 아주 편해요~
    memory = [[[initValue for _ in range(w)] for _ in range(h)] for _ in range(t)]
    return memory   # C 언어 계열을 제외하고는 메모리 해제를 해줄 필요가없음
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
def OnOpenDocument() :  #(2-2)
    global window, canvas, paper, inImage, outImage     #(4-1) global로 전역변수 설정해주자!
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    filename = askopenfilename(parent=window,
            filetypes=(("Color 이미지", "*.jpg;*.png;*.bim;*.tif"),("All files","*,*")) ) #(4-2)
    # (5-1) 파일 열었다가 취소 눌렀을 때, 파이참에는 오류가 나지만, 실제로는 티도안남 굳이 이걸 만들어 놓을 필요는 없음.
    if filename == '' or filename ==None:
        return
    # (4-3) 이미지 파일 --> OpenCV 객체
    inPhoto = cv2.imread(filename)
    # (4-4) 중요! 입력 이미지 크기를 파악
    inH = inPhoto.shape[0]
    inW = inPhoto.shape[1]
    # (5-2) 상태바에 파일이름 넣어주자
    sbar.configure(text=filename.split('/')[-1] + '  (' +str(inH) + 'x' + str(inW) + ')')  # filename.split('/') 는 리스트로 저장됨. 리스트 마지막
    #messagebox.showinfo('',str(inH) + 'x' + str(inW))
    # (4-6) 메모리 할당
    inImage = malloc3D(inH, inW, RGB)
    # (4-7) openCV 개체 --> 입력 이미지
    for i in range(inH):
        for k in range(inW):
            inImage[RR][i][k] = inPhoto.item(i,k,BB)
            inImage[GG][i][k] = inPhoto.item(i,k,GG)
            inImage[BB][i][k] = inPhoto.item(i,k,RR)
    equalImage()    #(4-8)
def equalImage() :  #(2-4)
    global window, canvas, paper, inImage, outImage     #(4-9) global로 전역변수 설정해주자!
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    # (4-10) 중요! 출력 이미지의 크기를 결정 ---> 알고리즘에 의존
    outH = inH
    outW = inW
    # (4-11) 메모리 할당
    outImage = malloc3D(outH, outW, RGB)
    ### (4-12) 진짜 영상처리 알고리즘 ###
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                outImage[rgb][i][k] = inImage[rgb][i][k]
    #########################
    OnDraw() # (4-13)
def OnDraw():   # (4-14)
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    if canvas != None : # 기존에 화면 출력한적 있는지 확인 -> 있으면 떼어버리자
        canvas.destroy()
    window.geometry(str(outW) + 'x' + str(outH)) # window 창 크기 조절 복사 붙여넣어서 사용하자!
    # (4-15) 캔버스, 페이퍼 생성 # 윈도우는 만들어뒀고, 칠판인 캔버스를 만들자, 그 후에 점찍을 수 있는 페이퍼 만들자!
    canvas = Canvas(window, height=outH, width=outW)    # 칠판 준비하기
    paper = PhotoImage(height=outH, width=outW)         # 칠판 크기같은 빈종이 준비하기
    #canvas.create_image()
    canvas.create_image((outW//2, outH//2), image = paper, state='normal') # 빈종이를 어디에 붙일래? 이제 점 찍을 차례
   # # (4-16) 출력 메모리 --> 화면에 찍기
   # for i in range(outH):
   #     for k in range(outW):
   #         r = outImage[RR][i][k]
   #         g = outImage[GG][i][k]
   #         b = outImage[BB][i][k]
   #         paper.put('#%02x%02x%02x' % (r,g,b),(k,i))
    rgbString = "" # 전체 화면에 찍을 내용을 메모리에 저장해 놓기
    #(4-18) 위에 for문 주석처리하고 새로운 for문
    for i in range(outH):
        tmpString = "" #  한 줄에 해당하는 내용
        for k in range(outW):
            r = outImage[RR][i][k]
            g = outImage[GG][i][k]
            b = outImage[BB][i][k]
            tmpString += '#%02x%02x%02x ' % (r,g,b)  #제일 뒤에 공백 1개
        rgbString += '{' + tmpString + '} '  #제일 뒤에 공백 1개
    paper.put(rgbString)
    # (4-17) 캔버스를 벽(winodw)에 붙이기
    canvas.pack(expand=1, anchor=CENTER) # 벽 가운데 붙이기
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


def apply_yolo():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename
    global cap, model

    if filename is None or filename == '':
        messagebox.showwarning("경고", "먼저 동영상 파일을 열어주세요.")
        return

    # YOLO 모델 로드
    model = YOLO('yolov8n.pt')

    # 동영상 열기
    cap = cv2.VideoCapture(filename)

    # # 캔버스 크기 설정   (alpha 2 삭제)
    # ret, frame = cap.read()
    # if ret:
    #     height, width = frame.shape[:2]
    #     canvas.config(width=width, height=height)

    update_frame()
def update_frame():
    ''' alpha 2에서 삭제
    global cap, canvas, model

    ret, frame = cap.read()
    if ret:
        # YOLO 적용
        results = model.track(frame, persist=True)
        frame_ = results[0].plot()

        # OpenCV BGR to RGB 변환
        rgb_frame = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)

        # PIL ImageTk 변환
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))

        # 캔버스에 이미지 표시
        canvas.create_image(0, 0, anchor=NW, image=photo)
        canvas.image = photo

        # 다음 프레임 업데이트 예약
        window.after(25, update_frame)
    else:
        cap.release()
    alpha 2에서 추가되 내용(밑에)
    '''
    global cap, canvas, model, window

    ret, frame = cap.read()
    if ret:
        # YOLO 적용
        results = model.track(frame, persist=True)
        frame_ = results[0].plot()

        # OpenCV BGR to RGB 변환
        rgb_frame = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)

        # 창 크기에 맞게 프레임 크기 조정
        window_width = canvas.winfo_width()
        window_height = canvas.winfo_height()
        rgb_frame = cv2.resize(rgb_frame, (window_width, window_height))

        # PIL ImageTk 변환
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))

        # 캔버스에 이미지 표시
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=NW, image=photo)
        canvas.image = photo

        # 다음 프레임 업데이트 예약
        window.after(25, update_frame)
    else:
        cap.release()
def OnOpenVideo():
    global window, canvas, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename

    filename = askopenfilename(parent=window,
                               filetypes=(("Video files", "*.mp4;*.avi"), ("All files", "*,*")))

    if filename == '' or filename == None:
        return

    # 상태바에 파일이름 넣어주기
    sbar.configure(text=filename.split('/')[-1])

    messagebox.showinfo('성공', f'{filename} 파일이 선택되었습니다.')

def on_resize(event):
    if cap is not None:
        update_frame()

# 전역 변수부
window, canvas, paper = None, None, None
inImage, outImage = [], []
inH, inW, outH, outW = [0] * 4
inPhoto, outPhoto = None, None
filename = None
RGB, RR, GG, BB = 3, 0, 1, 2
cap, model = None, None

## 메인 코드부
window = Tk()
window.title("YOLO Object Detection")

# 화면 크기 설정
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
window.geometry(f"{screen_width}x{screen_height}")

## 메뉴 생성
mainMenu = Menu(window)
window.config(menu=mainMenu)

fileMenu = Menu(mainMenu, tearoff=0)
mainMenu.add_cascade(label='파일', menu=fileMenu)
fileMenu.add_command(label='동영상 열기', command=OnOpenVideo)
fileMenu.add_command(label='저장', command=OnSaveDocument)
fileMenu.add_separator()
fileMenu.add_command(label='종료', command=OnCloseDocument)

yoloMenu = Menu(mainMenu, tearoff=0)
mainMenu.add_cascade(label='YOLO', menu=yoloMenu)
yoloMenu.add_command(label='YOLO 적용', command=apply_yolo)

# 캔버스 생성
canvas = Canvas(window)
canvas.pack(expand=YES, fill=BOTH)

# 상태바 추가
sbar = Label(window, text='준비중', bd=1, relief=SUNKEN, anchor=W)
sbar.pack(side=BOTTOM, fill=X)

window.mainloop()