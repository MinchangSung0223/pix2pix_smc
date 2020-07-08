import numpy as np   # for zeros
import cv2 as cv
import random        # for randrange
import scipy.interpolate as spi
from scipy.interpolate import BSpline

import numpy as np
import scipy.interpolate as si
def scipy_bspline(cv, n=100, degree=3, periodic=False):
       """ Calculate n samples on a bspline

           cv :      Array ov control vertices
           n  :      Number of samples to return
           degree:   Curve degree
           periodic: True - Curve is closed
       """
       cv = np.asarray(cv)
       count = cv.shape[0]

       # Closed curve
       if periodic:
           kv = np.arange(-degree,count+degree+1)
           factor, fraction = divmod(count+degree+1, count)
           cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)),-1,axis=0)
           degree = np.clip(degree,1,degree)

       # Opened curve
       else:
           degree = np.clip(degree,1,count-1)
           kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

       # Return samples
       max_param = count - (degree * (1-periodic))
       spl = si.BSpline(kv, cv, degree)
       return spl(np.linspace(0,max_param,n))


mouse_is_pressing = False   # 왼쪽 마우스 버튼 상태 체크를 위해 사용
drawing_mode = True       # 현재 그리기 모드 선택을 위해 사용 ( 원 / 사각형 )
start_x, start_y = -1, -1   # 최초로 왼쪽 마우스 버튼 누른 위치를 저장하기 위해 사용
color = (0, 0, 255)   # 도형 내부 채울때 사용할 색 지정시 사용 ( 초기값은 흰색 )
save_img =np.zeros((512, 512, 3), np.uint8) 
temp_img = np.zeros((512, 512, 3), np.uint8) 
pointList = []
thick = 4
def mouse_callback(event,x,y,flags,param):

	global color, start_x, start_y, drawing_mode, mouse_is_pressing,pointList

	if event == cv.EVENT_LBUTTONDOWN:
		# 랜덤으로 (blue, green, red)로 사용될 색을 생성
	
		mouse_is_pressing = True     # 왼쪽 마우스 버튼을 누른 것 감지 
		start_x, start_y = x, y     # 최초로 왼쪽 마우스 버튼 누른 위치를 저장 
		save_img= temp_img.copy()
		pointList.append([x,y])
		print(pointList)


img = np.zeros((512, 512, 3), np.uint8) 
cv.namedWindow('image')   
cv.setMouseCallback('image', mouse_callback) 


while(1):
	pointList_ = np.array(pointList,dtype=int)

	#print(pointList_.shape)
	try:
		xd = np.linspace(min(pointList_[:,0]), max(pointList_[:,0]), 1000)
		iyd = scipy_bspline(pointList_,n=1000,degree=5,periodic=False)
		temp_img = save_img.copy()
		for i in range(0,len(xd)):
			cv.circle(temp_img,(int(iyd[i,0]),int(iyd[i,1])),thick,color,-1)
	except:
		pass

	
	

	cv.imshow('image',temp_img)
	k = cv.waitKey(1) & 0xFF
	
	if k == ord('n'): # n 누르면 그리기 모드 변경
		drawing_mode = not drawing_mode
		save_img = temp_img.copy()
		pointList = []
	if k == ord('c'): # c
		if color == (255,0,0):
			color =(0,0,255)
		elif color == (0,0,255):
			color =(255,0,0)
	if k == ord('+'): # c
		if thick>1:
			thick = thick+1
	if k == ord('-'): # c
		if thick>1:
			thick = thick-1
	if k == ord('e'): # c
		save_img = np.zeros((512, 512, 3), np.uint8) 
		pointList = []

	elif k == 27: 
		new_img = np.concatenate((np.zeros((512, 512, 3), np.uint8) , save_img), axis=1)
		cv.imwrite('val2/testimage.png',new_img)
		break

cv.destroyAllWindows() 
