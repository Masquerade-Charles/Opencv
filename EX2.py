import cv2
import os
import numpy as np
from scipy.ndimage.filters import convolve
def get_energymap(img):  #得到能量图
    dx=np.array([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])  #x方向上sobel滤波器
    dx=np.stack([dx]*3,axis=2)  #2Dsobel滤波器变3Dsobel滤波器
    dy=np.array([[1.0,0.0,-1.0],[2.0,0.0,-2.0],[1.0,0.0,-1.0]])
    dy=np.stack([dy]*3,axis=2)
    img=img.astype('float32')  #转为浮点型
    convolved=np.absolute(convolve(img,dx))+np.absolute(convolve(img,dy))
    energymap=convolved.sum(axis=2)
    return energymap
def getM(img):
    energymap=get_energymap(img)
    M=energymap.copy()
    w,h=energymap.shape
    v1 = np.zeros(h,np.float)
    v1[0]=float(0x3f3f3f3f)
    v2 = np.zeros(h,np.float)
    v2[h-1]=float(0x3f3f3f3f)
    for i in range(1, w):
        v1[1:h]  = M[i-1, 0:h-1]
        v2[0:h-1] = M[i-1, 1: h]
        M[i] = np.min(np.stack([v1, v2, M[i-1]], axis=1), axis=1)+M[i]  # V1,V2,M[i-1] 分别对应i-1行的j-1，j+1，j，所以axix=1时没行的最小值正好对应上一行的三个值中最小的那个
    return M
def crave(img,N):
    E=get_energymap(img)
    for count in range(0,N):
        M=getM(img)
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
        img=img[MM].reshape(E.shape[0],E.shape[1]-1,3)
        E=get_energymap(img)
    return img
def main():
    img=cv2.imread("D:/image/1.jpg")
    cv2.namedWindow('Image')
    ans=crave(img,100)
    cv2.imwrite("D:\\image\map.jpg",ans)
    ans=cv2.imread("D:\\image\map.jpg")
    cv2.imshow('Image',ans)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()