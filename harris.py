import cv2
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
def getIx(Img):  #利用sobel算子得到Ix  
    Ix=np.matrix(Img)
    x=int(Img.shape[0])
    y=int(Img.shape[1])
    for i in range(x):
        for j in range(y):
            if i>=1 and j>=1 and i<x-1 and j<y-1:
                ans=abs(-int(Img[i-1,j-1])+int(Img[i+1,j-1])-2*(int(Img[i-1,j]))+2*(int(Img[i+1,j]))-int(Img[i-1,j+1])+int(Img[i+1,j+1]))
                if ans>255:
                    ans=255
                Ix[i,j]=ans
            else:
                Ix[i,j]=0
    return Ix
def getIy(Img):  #利用sobel算子得到Iy
    Iy=np.matrix(Img)
    x=int(Img.shape[0])
    y=int(Img.shape[1])
    for i in range(x):
        for j in range(y):
            if i>=1 and j>=1 and i<x-1 and j<y-1:
                ans=abs(-int(Img[i-1,j-1])-2*(int(Img[i,j-1]))-int(Img[i+1,j-1])+int(Img[i-1,j+1])+2*(int(Img[i,j+1]))+int(Img[i+1,j+1]))
                if ans>255:
                    ans=255
                Iy[i,j]=ans
            else:
                Iy[i,j]=0
    return Iy
def getall(Img):
    Ix=np.array(getIx(Img))
    Iy=np.array(getIy(Img))
    Sxx=Ix*Ix
    Syy=Iy*Iy
    Sxy=Ix*Iy
    Sxx=cv2.GaussianBlur(Sxx,(3,3),1.5)
    Syy=cv2.GaussianBlur(Syy,(3,3),1.5)
    Sxy=cv2.GaussianBlur(Sxy,(3,3),1.5)
    return Sxx,Syy,Sxy
def getR(Img):
    Sxx,Syy,Sxy=getall(Img)
    x=Img.shape[0]
    y=Img.shape[1]
    R=np.zeros(Img.shape)
    for i in range(x):
        for j in range(y):
            M=[[Sxx[i,j],Sxy[i,j]],[Sxy[i,j],Syy[i,j]]]
            R[i,j]=np.linalg.det(M)-0.06*((np.trace(M)*(np.trace(M))))
    return R
def getAngularpoint(Img,img_c):
    R=getR(Img)
    R=np.abs(R)
    x=int(Img.shape[0])
    y=int(Img.shape[1])
    maxR=np.max(R)*0.5
    for i in range(2,x-2):
        for j in range(2,y-2):
            Rm=np.copy(R[i-2:i+3,j-2:j+3])
            Rm[2,2]=0
            if R[i,j]>np.max(Rm) and R[i,j]>maxR:
                cv2.circle(img_c,(j,i),1,(255,255,255),0)
    return img_c
def main():
    img=cv2.imread("D:/image/2.jpg")
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('Image')
    ans=getAngularpoint(gray,img)
    cv2.imwrite("D:\\image\sup.jpg",ans)
    ans=cv2.imread("D:\\image\sup.jpg")
    cv2.imshow('Image',ans)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()
