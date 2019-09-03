import tkinter as tk 
from tkinter import filedialog
from PIL import Image
import pytesseract as pyt 
import numpy as np
import cv2 as cv
import sys
import imutils 
from matplotlib import pyplot as plt 
import matplotlib.image as mpimg


root=tk.Tk()
root.withdraw()
file=filedialog.askopenfilename()
 
img=cv.imread(file)
imgR=imutils.resize(img,width=500)
original=imgR.copy()
imgR=cv.bilateralFilter(imgR.copy(),9,15,15)



newImg=np.zeros(imgR.shape,imgR.dtype)

try:
    alpha=float(input("Enter the alpha vlaue [1.0-3.0]: "))
    beta=int(input("Enter the beta vlaue [0-100]: "))
except ValueError:
    print("Error , not a number")

for y in range(imgR.shape[0]):
    for x in range(imgR.shape[1]):
        for c in range(imgR.shape[2]):
            newImg[y,x,c]=np.clip(3*imgR[y,x,c],0,255)

newImg=cv.bilateralFilter(newImg.copy(),7,15,15)



hsv=cv.cvtColor(newImg,cv.COLOR_BGR2HSV)

# For white
lower_white = np.array([127,59,180], dtype=np.uint8)
upper_white = np.array([130,68,255], dtype=np.uint8)

imgW=cv.inRange(hsv,lower_white,upper_white)
#Guilt and regret have killed many a man before their time.
# For yellow
lower_yellow = np.array([20, 249, 64], dtype=np.uint8)
upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

imgY=cv.inRange(hsv,lower_yellow,upper_yellow)

combinedImage=cv.bitwise_or(imgW,imgY)
cv.imshow("combined Image",combinedImage)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite("intermediate.jpg",combinedImage)

text=pyt.image_to_string(newImg)
time.sleep(5)

print("Text extracted :",text)

h,w=combinedImage.shape
z=int(input("Enter the size of kernel"))
rectKernel=cv.getStructuringElement(cv.MORPH_RECT,(15,15))
tophat=cv.morphologyEx(combinedImage,cv.MORPH_TOPHAT,rectKernel)

cv.imshow('after filtering',tophat)
cv.waitKey(0)
cv.destroyAllWindows()

contours, hierarchy= cv.findContours(tophat.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

contours=sorted(contours, key = cv.contourArea, reverse = True)[:10]
print("length of contours ",len(contours))

NumPltCnt=[]
for c in contours:
        epsilon =0.02* cv.arcLength(c, True)
        approx = cv.approxPolyDP(c,epsilon,True)
        if len(approx) == 4: 
            NumPltCnt.append(approx) 

print("Number of rectangles found :",len(NumPltCnt))
print(NumPltCnt) 

if(len(NumPltCnt)==0):
    sys.exit("No Number plate found !!.")


original=cv.drawContours(original,NumPltCnt,-1,(0,255,0),3)

cv.imshow("Final",original)
cv.waitKey(0)
cv.destroyAllWindows()