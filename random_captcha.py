#coding=utf-8
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.misc import imread,imresize,imsave
import string

#用于生成不同长度的训练集（3..6）/验证集
characters=string.digits+string.ascii_uppercase+string.ascii_lowercase
width,height,n_len,n_class=170,80,4,len(characters)

generater=ImageCaptcha(width=width,height=height)
dir="/home/jobs/Desktop/code/captcha/training"
for i in range(10000):
	random_str="".join([random.choice(characters) for j in range(0,random.choice(range(3,6)))])
	image=generater.generate_image(random_str)	
	imsave(dir+"/"+random_str+".jpg",image)
	
