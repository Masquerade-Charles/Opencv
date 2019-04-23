import cv2
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
import os
import math
svmpeopel="D:/image/svmpeopel"
svmcar="D:/image/svmcar"
testpeople="D:/image/testpeople"
testcar="D:/image/testcar"
cn=15
def gray(Img):
    return Img[:,:,0]*0.299+Img[:,:,1]*0.587+Img[:,:,0]*0.144
def getdx(Img):#得到x方向上的梯度
    fx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float)
    return convolve(Img,fx)
def getdy(Img):#得到y方向上的梯度
    fy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]],np.float)
    return convolve(Img,fy)
def Gamma(Img,gamma):#归一化
    return np.exp(np.log(Img/np.max(Img))*gamma)*255
def gaussfilter(Img,size,sigma):#用高斯滤波去噪
    PI=math.acos(-1)
    f=np.zeros((size,size),np.float)
    for i in range(size):
        for j in range(size):
            f[i][j]=(i-size//2)*(i-size//2)+(j-size//2)*(j-size//2)
    f=1.0/(2.0*PI*sigma*sigma)*np.exp(-f/(2.0*sigma*sigma))
    return convolve(Img,f) 
def blockgamma(Img,sigma):#块归一化
    ans=Img/np.sqrt(Img.T*Img+sigma*sigma)
    return ans
def exact(angle,c_size,k_size,bin=9,sigma=0.1):#分块与合并
    h,w=c_size
    H,W=angle.shape
    ht=np.zeros((H//h,W//w,bin),np.float)
    for i in range(H//h):
        for j in range(W//w):
            t,x=np.histogram(angle[i*h:i*h+h,j*w:j*w+w],bin,(-180,180))
            ht[i][j]=t
    H=H//h
    W=W//w
    h,w=k_size
    l=h*w*bin
    res=np.zeros((H//h,W//w,l),np.float)
    for i in range(H//h):
        for j in range(W//w):
            res[i][j]=blockgamma(np.reshape(ht[i*h:i*h+h,j*w:j*w+w],(l)),sigma)
    return np.reshape(res,(1,res.shape[0]*res.shape[1]*res.shape[2]))
def gethog(Img):
    PI=math.acos(-1)
    Img=np.array(Img,np.float)
    Img=Gamma(Img,1)
    Img=gray(Img)
    Img=gaussfilter(Img,7,1.0)
    dx=getdx(Img)
    dy=getdy(Img)
    angle=1/PI*180*np.arctan(dy/dx)
    hog=exact(angle,(8,8),(2,2))
    return hog
def train(path,c1):
    c=0
    X=0
    for filename in os.listdir(path):
        if c>=cn:
            break
        if filename.endswith('.jpg'):
            filename=path+'/'+filename
            img=cv2.imread(filename)
            hog=gethog(img)
            if c==0:
                X=hog
            else:
                X=np.vstack([X,hog])
            c+=1
    Y=np.ones((cn),np.float)*c1
    return X,Y
if __name__ == "__main__":
    X,Y=train(svmpeopel,0)
    x,y=train(svmcar,1)
    X=np.vstack([X,x])
    Y=np.hstack([Y,y])
    tr=svm.SVC()
    tr.fit(X,Y)
    cn=5
    X_,Y_=train(testpeople,0)
    x_,y_=train(testcar,1)
    X_=np.vstack([X_,x_])
    Y_=np.hstack([Y_,y_])
    result=tr.predict(X_)
    t=Y_-result
    print(t)
    t=np.where(t==0,1,0)
    d=1
    for i in t.shape:
        d*=i
    print(100*np.sum(t)/d,"%")