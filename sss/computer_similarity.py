import sys, random, os
sys.path.insert(0, '../../python')
import mxnet as mx
import numpy as np
import triplet_loss , cv2
import python_surf
from operator import itemgetter
class Similarity(object):
    def __init__(self,top_similarity_img_want,imgs,baseName,names,savemodel,imagerootdir,epoch_num,batch_size=1):
        self.savemodel = savemodel
        self.root = imagerootdir
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.names = names
        self.baseName = baseName
        self.imgs = imgs
        self.top_similarity_img_want = top_similarity_img_want
        self.baseNameIdx = -1

    def init_net(self):
        _, arg_params, __ = mx.model.load_checkpoint(self.savemodel, 20)
        network = triplet_loss.get_sim_net()

        input_shapes = dict([('same', (self.batch_size, 3, 32, 32)), \
                             ('diff', (self.batch_size, 3, 32, 32))])
        executor = network.simple_bind(ctx=mx.cpu(), **input_shapes)
        for key in executor.arg_dict.keys():
            if key in arg_params:
                print(key, arg_params[key].shape, executor.arg_dict[key].shape)
                arg_params[key].copyto(executor.arg_dict[key])
        return executor

    def Mat2Npy(self):
        for fn in os.listdir(self.root):
            if fn.endswith(".jpg"):
                compath = self.root + "/" + fn
                if fn != self.baseName:
                    #与基准图进行对齐
                    mysurf = python_surf.surf_(self.root + "/" + self.baseName,compath)
                    image_,_,_ = mysurf.suftImageAlignment()
                    fn,suffix = os.path.splitext(fn)
                    image = cv2.resize(image,(32,32))
                    pynpath = self.root + "/" + fn
                    np.save(pynpath,image)
                else:
                    image = cv2.imread(compath)
                    fn, suffix = os.path.splitext(fn)
                    image = cv2.resize(image, (32, 32))
                    pynpath = self.root + "/" + fn
                    np.save(pynpath, image)
    def getImages(self):
        self.Mat2Npy()
        for fn in os.listdir(self.root):
            if fn.endswith(".npy"):
                self.names.append(self.root + "/" + fn)
        random.shuffle(self.names)
        fnjpg,suffixjpg = os.path.splitext(self.baseName)
        for i in range(0,len(self.names)):
            fnpyn,suffixpyn = os.path.splitext(self.names[i])
            pos = fnpyn.rfind("/",0,)
            fnpyn = fnpyn[pos+1:len(fnpyn)]
            if fnpyn == fnjpg:
                self.baseNameIdx = i
            self.imgs.append(np.load(self.names[i]))
        if self.baseNameIdx == -1:
            print("基准图片文件名选择出错，请重新选择基准图片 \n")
            return -1
        return 0
    def save_img(self,fname,im):
        a = np.copy(im)*255.0
        b = a.transpose(0,1,2)
        cv2.imwrite(fname,b)

    def fix_error(self,src):
        c = []
        for i in range(32):
            c.append(src.tolist())
        c = np.array(c)
        c.reshape(32,32,3)
        return c

    def computer_similarity(self):
        executor = self.init_net()
        ret = self.getImages()
        if ret == -1:
            return ret
        imagenum = len(self.names)
        #这儿只选择了img的随机一行进行相似度的比较
        #后期可以考虑选择多行，进行相似度比较
        src = self.imgs[self.baseNameIdx][random.randint(0,len(self.imgs[self.baseNameIdx])-1)]
        src = self.fix_error(src)
        dsts = []
        for i in range(imagenum):
            for j in range(128):
                k = random.randint(0,len(self.imgs[i]) -1)
                dst = self.imgs[i][k]
                dst = self.fix_error(dst)
                outputs = executor.forward(is_train=True, same=mx.nd.array([src]).reshape(1, 3, 32, 32),
                                           diff=mx.nd.array([dst]).reshape(1, 3, 32, 32))
                dis = outputs[0].asnumpy()[0]
                dsts.append((dst, dis, i))
        imageyouwant = dict()
        i = 0
        for img,w,la,in sorted(dsts,key=itemgetter(1))[:20]:
            if i<len(self.names):
                if la not in imageyouwant:
                    fnpyn,suffixpyn = os.path.splitext(self.names[la])
                    imageyouwant[la] = cv2.imread(fnpyn+".jpg")
                    i +=1
                    if i == self.top_similarity_img_want:
                        break
        return imageyouwant

