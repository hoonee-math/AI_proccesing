'''
메뉴를 예쁘게 만들어보기 위한 알파3
'''

from tkinter import *    #(1-3)
from tkinter import ttk
from tkinter import messagebox
from tkinter.filedialog import *
from tkinter.simpledialog import *
import math
import os.path #파일 처리 / 다음부터는 복붙
import cv2      #(3-3)
import numpy as np
from tkinter import messagebox, filedialog
# from PyQt5.QtWidgets import QMainWindow, QApplication, QToolButton   # https://velog.io/@hj8853/Python-PyQt5-%EA%B3%84%EC%82%B0%EA%B8%B0-%EB%A7%8C%EB%93%A4%EA%B8%B0

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
def OnOpenDocument():
    global filename, inPhoto, inImage, outImage, inH, inW, outH, outW
    filename = filedialog.askopenfilename(
        filetypes=(("이미지/비디오 파일", "*.jpg;*.png;*.bmp;*.tif;*.mp4;*.avi;*.mov"),
                   ("모든 파일", "*.*")))

    if filename == '' or filename is None:
        return

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
    elif file_extension in ['mp4', 'avi', 'mov']:
        # 비디오 처리 코드...
        create_video_menu()

    sbar.configure(text=filename.split('/')[-1])
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
def OnDraw():
    global window, canvas, paper, inImage, outImage
    global inH, inW, outH, outW, inPhoto, outPhoto, filename, canvas_frame

    if canvas != None:
        canvas.destroy()

    canvas = Canvas(canvas_frame, height=outH, width=outW)
    paper = PhotoImage(width=outW, height=outH)
    canvas.create_image((outW//2, outH//2), image=paper, state='normal')

    rgbString = ""
    for i in range(outH):
        tmpString = ""
        for k in range(outW):
            r = outImage[RR][i][k]
            g = outImage[GG][i][k]
            b = outImage[BB][i][k]
            tmpString += "#%02x%02x%02x " % (r, g, b)
        rgbString += "{" + tmpString + "} "
    paper.put(rgbString)

    canvas.pack(expand=1, anchor=CENTER)

## 메뉴 관련 함수 ##
def create_image_menu():
    global menu_frame
    for widget in menu_frame.winfo_children():
        widget.destroy()

    Button(menu_frame, text="파일 열기", command=OnOpenDocument).pack(fill=X, padx=10, pady=5)
    # Add a separator
    ttk.Separator(menu_frame, orient=HORIZONTAL).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="동일 이미지", command=equalImage).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="밝게/어둡게", command=addImage).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="그레이스케일", command=grayImage).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="반전", command=reversedImage).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="확대/축소", command=zoomImage).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="이동", command=moveImage).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="엠보싱(RGB)", command=embossImageRGB).pack(fill=X, padx=10, pady=5)
def create_video_menu():
    global menu_frame
    for widget in menu_frame.winfo_children():
        widget.destroy()

    Button(menu_frame, text="파일 열기", command=OnOpenDocument).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="재생", command=play_video).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="일시정지", command=pause_video).pack(fill=X, padx=10, pady=5)
    Button(menu_frame, text="정지", command=stop_video).pack(fill=X, padx=10, pady=5)
    # 비디오 관련 추가 버튼들...
def create_initial_menu():
    global menu_frame
    Button(menu_frame, text="파일 열기", command=OnOpenDocument).pack(fill=X, padx=10, pady=5)  # 첫 화면의 왼쪽 파일 열기 버튼 padx, pady : 여백
    # 다른 초기 버튼들을 여기에 추가할 수 있습니다.

## 이미지 처리 함수
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

# 전역 변수부
window, canvas, paper = None, None, None    #(1-1)
inImage, outImage = [], []      #(3-1) unsigned char **m_inImage.... 영상처리를 위한 전역변수 선언!
inH, inW, outH, outW = [0]*4
inPhoto, outPhoto = None, None      #(3-2) OpenCV를 사용하려면 설치해야함(p38)
filename = None
RGB, RR, GG, BB = 3, 0, 1, 2    #(4-5)

## 메인 코드부
window = Tk()   #(1-2)
window.title("AI 영상인식 (Alpha_1)")    #(1-5)
window.geometry("800x600")  # *로 곱하면 오류

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

# 초기 메뉴 버튼 생성
create_initial_menu()

# 메뉴 프레임 생성
menu_frame = Frame(window, width=200)
menu_frame.pack(side=LEFT, fill=Y)

# 캔버스 프레임 생성
canvas_frame = Frame(window)
canvas_frame.pack(side=RIGHT, expand=True, fill=BOTH)

# # 파일 메뉴
# mainMenu = Menu(window)
# window.config(menu=mainMenu)
# fileMenu = Menu(mainMenu, tearoff=0)
# mainMenu.add_cascade(label='파일', menu=fileMenu)
# fileMenu.add_command(label='열기', command=OnOpenDocument)
# fileMenu.add_command(label='저장', command=OnSaveDocument)
# fileMenu.add_separator()
# fileMenu.add_command(label='종료', command=window.quit)

window.mainloop()