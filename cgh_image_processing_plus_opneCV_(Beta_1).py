from tkinter import *                       #(1-3)
from tkinter import messagebox
from tkinter.filedialog import *
from tkinter.simpledialog import *
import math
import cv2      #(3-3)
import numpy as np
import os.path #파일 처리 / 다음부터는 복붙
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

### 영상 처리 함수 ###
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

#### Open CV 함수부 ####

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
    ## 화면에 찾은 네모 그리기
    for (x, y, w, h) in face_rects :
        cv2.rectangle(outPhoto,(x,y),(x+w, y+h),(0,255,0),0)
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
image1Menu.add_command(label='흑백', command=bwImage)
image1Menu.add_command(label='포스터라이징', command=posterImage)
image2Menu = Menu(mainMenu, tearoff=0)
mainMenu.add_cascade(label='기하학 처리', menu=image2Menu)
image2Menu.add_command(label='대칭', command=mirrorImage)
image2Menu.add_command(label='확대/축소', command=zoomImage)
image2Menu.add_command(label='이동', command=moveImage)
image2Menu.add_command(label='회전', command=rotateImage)
image3Menu = Menu(mainMenu, tearoff=0)
mainMenu.add_cascade(label='화소 영역 처리', menu=image3Menu)
image3Menu.add_command(label='엠보싱(RGB)', command=embossImageRGB)
image3Menu.add_command(label='엠보싱(HSV)', command=embossImageHSV)
image3Menu.add_command(label='블러링(RGB)', command=blurImageRGB)
image3Menu.add_command(label='블러링(HSV)', command=blurImageHSV)
image3Menu.add_command(label='샤프닝(RGB)', command=sharpRGB)
image3Menu.add_command(label='샤프닝(HSV)', command=sharpHSV)
image3Menu.add_command(label='경계선(RGB)', command=lineRGB)
image3Menu.add_command(label='경계선(HSV)', command=lineHSV)
image3Menu.add_command(label='가우시안(RGB)', command=gaussianRGB)
image3Menu.add_command(label='가우시안(HSV)', command=gaussianHSV)
openCVMenu = Menu(mainMenu, tearoff=0)
mainMenu.add_cascade(label='Open CV 처리', menu=openCVMenu)
openCVMenu.add_command(label='동일 이미지', command=equalImageCV)
openCVMenu.add_command(label='그레이 이미지', command=grayscaleImageCV)
openCVMenu.add_command(label='HSV 변환', command=hsvImageCV)
openCVMenu.add_command(label='binary 이미지', command=binaryImageCV)
openCVMenu.add_command(label='적응형 이미지', command=binary2ImageCV)
openCVMenu.add_command(label='엠보싱 이미지', command=embossImageCV)
openCVMenu.add_command(label='카툰 이미지', command=cartonImageCV)
openCVMenu.add_separator()
openCVMenu.add_command(label='얼굴인식', command=faceDetectCV)
openCVMenu.add_command(label='코인식', command=noseDetectCV)

window.mainloop()   #(1-4)