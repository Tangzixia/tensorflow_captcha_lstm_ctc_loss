#coding=utf-8
import numpy as np
from scipy.misc import imread,imresize
import cv2
import os

#用于生成对应的dict映射文件,['0'-'9','A'-'Z','a'-'z']分别与[1-62]一一映射
def write_char(file,list=[i for i in range(48,58)]+[i for i in range(65,91)]+[i for i in range(97,123)]):
    with open(file,'w') as f:
        count=0
        for item in list:
            f.write(str(count)+":"+chr(item)+"\n")
            count+=1
def get_dict(file="dict.txt"):
    dict_ = {}
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            count, label = line.split("\n")[0].split(":")
            dict_[label] = int(count)
    return dict_
def random_batch(batch_size,dir_path="testing"):
    batches_imgs=[]
    batches_indices=[]
    batches_values=[]
    batches_shapes=[]
    dict_=get_dict()


    batch_indices=[]
    batch_values=[]
    batch_imgs=[]
    files=os.listdir(dir_path)
    batch_count=0
    for file in files:
        path=os.path.join(dir_path,file)
        img = imresize(imread(path), [170, 80])
        img = imresize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), [170, 80])
        label=file.split(".")[0]
        batch_imgs.append(img)
        for i,item in enumerate(label):
            batch_indices.append((batch_count,i))
            batch_values.append(dict_[item])
        batch_count+=1
        if batch_count%batch_size==0 and batch_count!=0:
            batch_count=0
            batches_imgs.append(np.asarray(batch_imgs,dtype=np.float32))
            batches_indices.append(np.asarray(batch_indices,dtype=np.int64))
            batches_values.append(np.asarray(batch_values,dtype=np.int32))
            batches_shapes.append(np.asarray([batch_size,np.asarray(batch_indices,np.int64).max(0)[1]+1],dtype=np.int64))

            batch_imgs=[]
            batch_indices=[]
            batch_values=[]

    return batches_imgs,batches_indices,batches_values,batches_shapes
if __name__=="__main__":
    write_char(file="dict.txt")







