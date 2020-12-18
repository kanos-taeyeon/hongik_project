import os
import lief
import glob
from math import *
from PIL import Image
import cv2
import time
import numpy as np

def section(filename):

    #바이너리 전체를 불러온다.
    binaryValues = []
    binary = open(filename, 'rb').read()
    binaryvalues = list(binary)
	
    return binaryvalues	      

def text_section_image(data, file):
	
	x = 0
	#print(len(data))
	size = len(data) #데이터 크기
	print(size)
	W = ceil(sqrt(size)) #이미지 한 변의 길이
	h = int(W)
	K = W * W     	

	if(W == 0 or h == 0): #뽑아낸 데이터가 아무것도 없을 때
		print("Unable to create image because there is no data")
		
	if size > 50176 and size < K: #파일 데이터 크기가 이미지 한변의 길이의 제곱보다 작을 경우 남은 부분은 0byte로 처리 한다.

		image = Image.new('L',(h,h))
		print(image)

		space = K - size
		
		while x < space: #남은 부분은 전부 0으로 처리 
			data += [0]
			x = x + 1			
		image.putdata(data)

		imagename = file+".png"
		image.save(imagename)
		src = cv2.imread(imagename, cv2.IMREAD_GRAYSCALE)
		resize_image = cv2.resize(src, dsize=(224, 224), interpolation = cv2.INTER_AREA)
		cv2.imwrite(imagename, resize_image)		

		return 0

	elif size < 50176 and size < K:

		image = Image.new('L',(h,h))
		print(image)
		resize_image = image.resize((224, 224), Image.ANTIALIAS)

		space = K - size
		
		while x < space: #남은 부분은 전부 0으로 처리 
			data += [0]
			x = x + 1			
		resize_image.putdata(data)

		imagename = file+".png"
		resize_image.save(imagename)
		return 0

	elif size >= 50176 and size == K: 
		
		image = Image.new('L',(h,h))
		print(image)
		image.putdata(data)

		imagename = file+".png"
		image.save(imagename)
		src = cv2.imread(imagename, cv2.IMREAD_GRAYSCALE)
		resize_image = cv2.resize(src, dsize=(224 ,224), interpolation = cv2.INTER_AREA)
		cv2.imwrite(imagename, resize_image)	

		return 0

	elif size <= 50176 and size == K:

		image = Image.new('L',(h,h))
		print(image)
		resize_image = image.resize((224 ,224), Image.ANTIALIAS)
		resize_image.putdata(data)

		imagename = file+".png"
	
		resize_image.save(imagename)
		return 0

if __name__=="__main__":

	files = sorted(glob.glob('../datasets/packingtest/*'))
	#files = glob.glob('../Desktop/oav/*')
	count = 0
	for file in files:
           if count < 10000:
             with open(file, 'r') as f:
                  print(file)
             file_full_path=file #directory file route name
             path=os.path.dirname(file_full_path) # file route road
             base_name=os.path.splitext(os.path.basename(file_full_path))[0] #file name road
             #vir=os.path.splitext(os.path.basename(file_full_path))[-1]
             #base_name = base_name + vir
             #print(base_name)

             outputFilename=os.path.join("../images/test",base_name)
             #outputFilename=os.path.join("../Desktop/ova",base_name)
             binaryData=section(file_full_path)
		
             if binaryData == 0:
                print("not pe")
             else:
                text_section_image(binaryData, outputFilename)
             count = count + 1
           else:
             exit()
