from tkinter import *                       #(1-3)
from tkinter import messagebox
from tkinter.filedialog import *
from tkinter.simpledialog import *
import math
import os.path #파일 처리 / 다음부터는 복붙
import cv2      #(3-3)
import numpy as np
import _pyinstaller_hooks_contrib


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
window.title("AI 영상인식 (Alpha_1")    #(1-5)
window.geometry("400x200")  # *로 곱하면 오류

## 상태바 생성   #(1-6)
sbar = Label(window, text="상태바", bd=1, relief=SUNKEN, anchor=W)
sbar.pack(side=BOTTOM, fill=X)

## 메뉴 생성    #(2-1)
mainMenu = Menu(window)     # 메뉴의 틀
window.config(menu=mainMenu)

fileMenu = Menu(mainMenu, tearoff=0)   # 상위 메뉴(파일)
mainMenu.add_cascade(label='파일', menu=fileMenu)
fileMenu.add_command(label='열기', command=OnOpenDocument)    #(2-2)
fileMenu.add_command(label='저장', command=OnSaveDocument)
fileMenu.add_separator()
fileMenu.add_command(label='종료', command=OnCloseDocument)

image1Menu = Menu(mainMenu, tearoff=0)   # (2-3)
mainMenu.add_cascade(label='화소점 처리', menu=image1Menu)
image1Menu.add_command(label='동일 이미지', command=equalImage)  #(2-4)
image1Menu.add_command(label='밝게/어둡게', command=addImage)
image1Menu.add_command(label='그레이스케일', command=grayImage)
image1Menu.add_command(label='반전', command=reversedImage)

image2Menu = Menu(mainMenu, tearoff=0)
mainMenu.add_cascade(label='기하학 처리', menu=image2Menu)
image2Menu.add_command(label='확대/축소', command=zoomImage)
image2Menu.add_command(label='이동', command=moveImage)

image3Menu = Menu(mainMenu, tearoff=0)
mainMenu.add_cascade(label='화소 영역 처리', menu=image3Menu)
image3Menu.add_command(label='엠보싱(RGB)', command=embossImageRGB)
window.mainloop()   #(1-4)