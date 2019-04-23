import cv2
import tkinter.filedialog
from tkinter import *
from tkinter.simpledialog import askstring
import numpy as np
import os
from PIL import Image,ImageTk
from scipy.ndimage.filters import convolve
import matplotlib
from matplotlib import pyplot as plt
import math
class Test:
    myframe=''
    width=1200
    height=600
    swidth=0
    sheight=0
    alg=0
    path=""
    img=""
    img_png=""
    menubar=""
    label_img=""
    def __init__(self):
        self.myframe=Tk()
        self.swidth=self.myframe.winfo_screenwidth()
        self.sheight=self.myframe.winfo_screenheight()
        self.alg='%dx%d+%d+%d' % (self.width,self.height,(self.swidth-self.width)/2,(self.sheight-self.height)/2)
        self.myframe.geometry(self.alg)
        self.myframe.title("图像处理")
        self.myframe.resizable(width=False,height=False)
        self.menubar=Menu(self.myframe)
        self.addMenu()
    def openimg(self):
        self.img=Image.open(self.path)
        self.img_png=ImageTk.PhotoImage(self.img)
    def show(self,newimg):
        self.label_img=Label(self.myframe,image=newimg)
        self.label_img.grid(row=0,column=0)
    def openfile(self):
        self.path=tkinter.filedialog.askopenfilename()
        self.openimg()
        self.show(self.img_png)
    def zoom(self):#缩放
        Img=cv2.imread(self.path)
        size=Img.shape
        widthpx=float(askstring("比例变换","请输入你要缩放的横向比例:"))
        heightpx=float(askstring("比例变换","请输入你要缩放的纵向比例:"))
        res=cv2.resize(Img,(int(widthpx*size[1]),int(heightpx*size[0])),cv2.INTER_LINEAR)
        cv2.imwrite("D:\image\zoom.jpg",res)
        self.path="D:\image\zoom.jpg"
        self.openimg()
        self.show(self.img_png)
    def translation(self):#平移
        Img=cv2.imread(self.path)
        x=int(askstring("平移","请输入x轴平移的距离:"))
        y=int(askstring("平移","请输入y轴平移的距离:"))
        rows,cols=Img.shape[:2]
        H=np.float32([[1,0,x],[0,1,y]])
        res=cv2.warpAffine(Img,H,(cols,rows))
        cv2.imwrite("D:\image\\translation.jpg",res)
        self.path="D:\image\\translation.jpg"
        self.openimg()
        self.show(self.img_png)
    def Gray_scale_change(self):#灰度变换
        Img=cv2.imread(self.path)
        gray=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite("D:\image\Gray.jpg",gray)
        self.path="D:\image\Gray.jpg"
        self.openimg()
        self.show(self.img_png)
    def Gaussian_filter(self):#高斯滤波
        Img=cv2.imread(self.path)
        res=cv2.GaussianBlur(Img,(5,5),0)
        cv2.imwrite("D:\image\Gauss.jpg",res)
        self.path="D:\image\Gauss.jpg"
        self.openimg()
        self.show(self.img_png)
    def Median_filtering(self):#中值滤波
        Img=cv2.imread(self.path)
        res=cv2.cv2.medianBlur(Img, 5)
        cv2.imwrite("D:\image\Median.jpg",res)
        self.path="D:\image\Median.jpg"
        self.openimg()
        self.show(self.img_png)
    #去雾
    def zmMinFilterGray(self,src,r=7):#最小滤波值，即minI(y)
        return cv2.erode(src,np.ones((2*r-1,2*r-1)))
    def guidedfilter(self,I,p,r,eps):#使用导向滤波获得透射率图
        m_I=cv2.boxFilter(I,-1,(r,r))
        m_p=cv2.boxFilter(p,-1,(r,r))
        m_Ip=cv2.boxFilter(I*p,-1,(r,r))
        cov_Ip=m_Ip-m_I*m_p
        m_II=cv2.boxFilter(I*I,-1,(r,r))
        var_I=m_II-m_I*m_I
        a=cov_Ip/(var_I+eps)
        b=m_p-a*m_I
        m_a=cv2.boxFilter(a,-1,(r,r))
        m_b=cv2.boxFilter(b,-1,(r,r))
        return m_a*I+m_b
    def getV1(self,m,r,eps,w,maxV1):
        V1=np.min(m,2)#得到暗通道图
        V1=self.guidedfilter(V1,self.zmMinFilterGray(V1,7),r,eps)#得到透射率图
        bins=2000
        ht=np.histogram(V1,bins)    #将透射率图分成2000个亮度区间，得到概率
        d=np.cumsum(ht[0])/float(V1.size)#得到所有像素亮点的值的比例和
        for lmax in range(bins-1,0,-1):#按亮度大小取前0.1%的像素
            if d[lmax]<=0.999:
                break
        A=np.mean(m,2)[V1>=ht[1][lmax]].max()#求最大的亮点亮度
        V1=np.minimum(V1*w,maxV1)  #得到min（minI(y))
        return V1,A
    def deHaze(self,m,r=81,eps=0.001,w=0.95,maxV1=0.80):
        Y=np.zeros(m.shape)   #创建0矩阵
        V1,A=self.getV1(m,r,eps,w,maxV1)#得到V1,A
        for x in range(3):
            Y[:,:,x]=(m[:,:,x]-V1)/(1-V1/A)#t=1-V1/A,因为J=(I(x)-A)/t+A  J=(I(x)-A+At)/t  J=(I(x)-A+A(1-V1/A))/t  J=(I(x)-A(1-1+V1/A))/t    J=(I(x)-V1)/t
        return Y
    def defogging(self):#去雾
        Img=cv2.imread(self.path)
        ans=self.deHaze(Img/255.0)*255
        cv2.imwrite("D:\image\defogging.jpg",ans)
        self.path="D:\image\defogging.jpg"
        self.openimg()
        self.show(self.img_png)
    #seamcraving
    def get_energymap(self,Img):  #得到能量图
        dx=np.array([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])  #x方向上sobel滤波器
        dx=np.stack([dx]*3,axis=2)  #2Dsobel滤波器变3Dsobel滤波器
        dy=np.array([[1.0,0.0,-1.0],[2.0,0.0,-2.0],[1.0,0.0,-1.0]])
        dy=np.stack([dy]*3,axis=2)
        Img=Img.astype('float32')  #转为浮点型
        convolved=np.absolute(convolve(Img,dx))+np.absolute(convolve(Img,dy))
        energymap=convolved.sum(axis=2)
        return energymap
    def getM(self,Img):
        energymap=self.get_energymap(Img)
        M=energymap.copy()
        w,h=energymap.shape
        v1 = np.zeros(h,np.float)
        v1[0]=float(0x3f3f3f3f)
        v2 = np.zeros(h,np.float)
        v2[h-1]=float(0x3f3f3f3f)
        for i in range(1, w):
            v1[1:h]  = M[i-1, 0:h-1]
            v2[0:h-1] = M[i-1, 1: h]
            M[i] = np.min(np.stack([v1, v2, M[i-1]], axis=1), axis=1)+M[i]  # V1,V2,M[i-1] 分别对应i-1行的j-1，j+1，j，所以axix=1时每行的最小值正好对应上一行的三个值中最小的那个
        return M
    def crave(self,Img,N):
        E=self.get_energymap(Img)
        for count in range(0,N):
            M=self.getM(Img)
            loc=np.argmin(M[-1])
            MM=np.ones(M.shape, dtype=np.bool)
            for i in reversed(range(E.shape[0])):
                MM[i][loc]=False
                if i==0:
                    break
                if loc-1>=0 and (int)(M[i-1][loc-1])==int(M[i][loc]-E[i][loc]):
                    loc=loc-1
                elif loc+1<E.shape[1] and (int)(M[i-1][loc+1])==int(M[i][loc]-E[i][loc]):
                    loc=loc+1
                else:
                    loc=loc
            MM=np.stack([MM]*3,axis=2)
            Img=Img[MM].reshape(E.shape[0],E.shape[1]-1,3)
            E=self.get_energymap(Img)
        return Img
    def seamcrave(self):
        Img=cv2.imread(self.path)
        x=int(askstring("列次数","请输入您进行裁剪的列切数:"))
        ans=self.crave(Img,x)
        cv2.imwrite("D:\\image\seamcrave.jpg",ans)
        self.path="D:\\image\seamcrave.jpg"
        self.openimg()
        self.show(self.img_png)
    #Harris
    def getIx(self,Img):  #利用sobel算子得到Ix  
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
    def getIy(self,Img):  #利用sobel算子得到Iy
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
    def getall(self,Img):
        Ix=np.array(self.getIx(Img))
        Iy=np.array(self.getIy(Img))
        Sxx=Ix*Ix
        Syy=Iy*Iy
        Sxy=Ix*Iy
        Sxx=cv2.GaussianBlur(Sxx,(3,3),1.5)
        Syy=cv2.GaussianBlur(Syy,(3,3),1.5)
        Sxy=cv2.GaussianBlur(Sxy,(3,3),1.5)
        return Sxx,Syy,Sxy
    def getR(self,Img):
        Sxx,Syy,Sxy=self.getall(Img)
        x=Img.shape[0]
        y=Img.shape[1]
        R=np.zeros(Img.shape)
        for i in range(x):
            for j in range(y):
                M=[[Sxx[i,j],Sxy[i,j]],[Sxy[i,j],Syy[i,j]]]
                R[i,j]=np.linalg.det(M)-0.06*((np.trace(M)*(np.trace(M))))
        return R
    def getAngularpoint(self,Img,img_c):
        R=self.getR(Img)
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
    def harris(self):
        Img=cv2.imread(self.path)
        gray=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
        ans=self.getAngularpoint(gray,Img)
        cv2.imwrite("D:\\image\harris.jpg",ans)
        self.path="D:\\image\harris.jpg"
        self.openimg()
        self.show(self.img_png)
    def addMenu(self):
        filemenu=Menu(self.menubar,tearoff=0)
        filemenu.add_command(label="选择文件",command=self.openfile)
        self.menubar.add_cascade(label="文件",menu=filemenu)
        filemenu=Menu(self.menubar,tearoff=0)
        filemenu.add_command(label="缩放",command=self.zoom)
        filemenu.add_command(label="平移",command=self.translation)
        filemenu.add_command(label="灰度变换",command=self.Gray_scale_change)
        filemenu.add_command(label="高斯滤波",command=self.Gaussian_filter)
        filemenu.add_command(label="中值滤波",command=self.Median_filtering)
        filemenu.add_command(label="去雾",command=self.defogging)
        filemenu.add_command(label="图像裁剪",command=self.seamcrave)
        filemenu.add_command(label="Harris检测",command=self.harris)
        self.menubar.add_cascade(label="操作",menu=filemenu)
        self.myframe['menu']=self.menubar
if __name__ == "__main__":
    test=Test()
    test.myframe.mainloop()

