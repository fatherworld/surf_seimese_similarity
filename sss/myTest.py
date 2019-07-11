import sys,random,os
sys.path.insert(0,"../../python")
import mxnet as mx
import numpy as np
import cv2
import computer_similarity
top_similarity_img_want = 5
imgs = []
imagerootdir = "71"
baseName = "1.jpg"
names = []
savemodel = "./model/bayes"
epoch_num = 20
mysimilarity = computer_similarity.Similarity(top_similarity_img_want,imgs,baseName,names,savemodel,imagerootdir,epoch_num)
imageyouwant = mysimilarity.computer_similarity()
for value in imageyouwant.values():
    cv2.imshow("value",value)
    cv2.waitKey(1000)