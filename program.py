import tkinter as tk 
from tkinter import filedialog
from PIL import Image
import pytesseract as pyt 
import numpy as np
import cv2 as cv
import sys
import imutils 

root=tk.Tk()
root.withdraw()
file=filedialog.askopenfilename()
 
img=cv.imread(file)
imgR=imutils.resize(img,width=500)
original=imgR.copy()

cv.imshow("Resized image after loading", imgR)
cv.waitKey(0)
cv.destroyAllWindows()


newImg=np.zeros(imgR.shape,imgR.dtype)

contrast=float(input("Enter the value of color boosting[1-3]: "))
for y in range(imgR.shape[0]):
    for x in range(imgR.shape[1]):
        for c in range(imgR.shape[2]):
            newImg[y,x,c]=np.clip(contrast*imgR[y,x,c],0,255)

#newImg=cv.bilateralFilter(newImg.copy(),7,15,15)

cv.imshow('after increasing color intensity',newImg)
cv.waitKey(0)
cv.destroyAllWindows()

hsv=cv.cvtColor(newImg,cv.COLOR_BGR2HSV)

# Range of white color.
lower_white = np.array([0,0,180], dtype=np.uint8)
upper_white = np.array([180,38,255], dtype=np.uint8)

imgW=cv.inRange(hsv,lower_white,upper_white)
cv.imshow("white",imgW)
cv.waitKey(0)
cv.destroyAllWindows()

# Range of yellow color.
lower_yellow = np.array([20, 39, 64], dtype=np.uint8)
upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

imgY=cv.inRange(hsv,lower_yellow,upper_yellow)

cv.imshow("yellow",imgY)
cv.waitKey(0)
cv.destroyAllWindows()


combinedImage=cv.bitwise_or(imgW,imgY)
cv.imshow("combined Image",combinedImage)
cv.waitKey(0)
cv.destroyAllWindows()



ker=int(input("Enter the size of kernel[7-15] :"))
rectKernel=cv.getStructuringElement(cv.MORPH_RECT,(ker,ker))
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



box=[]
for c in NumPltCnt:
    rect=cv.minAreaRect(c)
    x=cv.boxPoints(rect)
    box.append(np.int0(x))
    #img=cv.drawContours(img,[box],0,(0,255,0),3)
print(box[0])
xCord=[]
yCord=[]
for x in box[0]:
    xCord.append(x[0])
    yCord.append(x[1])

xmin=min(xCord)
xMax=max(xCord)
ymin=min(yCord)
ymax=max(yCord)

croped=original[ymin:ymax,xmin:xMax]
amtOfTxt=tophat[ymin:ymax,xmin:xMax]

text=pyt.image_to_string(amtOfTxt,lang='eng')
print("The extracted text of number plate is:  ",text)

def isMaxWhite(plate):
    avg = np.mean(plate)
    if (avg >= 115):
        return True
    else:
        return False

if(isMaxWhite(amtOfTxt)):
    cv.imshow("Cropped",croped)
    cv.waitKey(0)
    cv.destroyAllWindows()




original=cv.drawContours(original,NumPltCnt,-1,(0,255,0),3)

cv.imshow("Final",original)
cv.waitKey(0)
cv.destroyAllWindows()

