import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import pickle
import os
from os import listdir, getcwd
from os.path import join
import pdb
import json
import cv2
from util_wsy import createPath,iouLT
import math
import shutil
import random
sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# classes = ["plane","drone"]
classes = ["in_helmet","no_helmet"]
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('data/wp/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('data/wp/VOCdevkit/VOC%s/wsy_labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        bb = convert((w,h), b)
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        # ------------- wsy  ------------------
        # b = (cls_id, float(xmlbox.find('xmax').text),  float(xmlbox.find('ymax').text),float(xmlbox.find('xmin').text),float(xmlbox.find('ymin').text))
        out_file.write(" ".join([str(a) for a in b]) + '\n')

def is_equal(x,y):
    if x['id']==y['id']:
        return x

def loadJson():
    
    outpath = createPath('/home/lab601/project/yolo_v3/darknet/data/gp/voc/VOCdevkit/')
    labelpath = createPath('/home/lab601/project/yolo_v3/darknet/data/gp/voc/VOCdevkit/labels/')
    imgpath = '/home/lab601/project/yolo_v3/darknet/data/gp/gp_link/val_new/'

    fid = open('/home/lab601/project/yolo_v3/darknet/data/gp/val.json',encoding='utf-8')
    data=json.load(fid) 
    imgFile = open(outpath+'val.txt', 'w')
    pdb.set_trace()
    wds = []
    hts = []
    for c in data['categories']:
        print(c['name'])
    # for d in data['images']:
    #     print(len(data['images']),d['id'])
    #     annos=filter(lambda x: x['image_id']==d['id'], data['annotations'])
    #     annos = [a for a in annos]
    #     if len(annos)==0:
    #         print('annos None:',d['id'])
    #         continue
    #     anno = annos[0]

    #     b = anno['bbox'] 
    #     image_id = d['file_name'].split('.')[0]
    #     bb = convert((d['width'],d['height']),[b[0],b[2]+b[0],b[1],b[3]+b[1]])

    #     labelFile = open(labelpath+image_id+'.txt', 'w')
    #     labelFile.write(str(anno['category_id'])+" "+" ".join([str(a) for a in bb]) + '\n')
    #     labelFile.close()

    #     imgFile.write(outpath+'JPEGImages/v'+d['file_name']+ '\n')
    imgFile.close()



def cmpCSV(cfile,file):
    import csv
    import numpy as np
    import os,shutil
    path = 'data/'
    fid1 = csv.reader(open(cfile,'r'))
    fid2 = csv.reader(open(file,'r'))
    cl = 1
    cdata = {}
    data = {}
    for cline,line  in zip(fid1,fid2):
        if cl:
            cl = 0
            continue
        cdata[cline[0]] = {'box':cline[1:-1],'category':int(cline[-1])}
        data[line[0]] = {'box':line[1:-1],'category':int(line[-1])}

    WCT = []
    IOU = []
    for key in cdata:
        print(key)
        # pdb.set_trace()
        boxA = [ int(float(i)) for i in cdata[key]['box']]
        boxB = [ int(i) for i in data[key]['box']]
        print('boxA:',cdata[key]['box'])
        print('boxA:',boxA)
        if cdata[key]['category'] != data[key]['category']:
            shutil.copyfile(path+'test_out/'+key,createPath(path+'WCT/')+key)
            WCT.append(key)
            continue
        data[key]['iou'] = iouLT([boxA[0],boxA[1],boxA[4],boxA[5]],[boxB[0],boxB[1],boxB[4],boxB[5]])
        IOU.append(data[key]['iou'])
        if data[key]['iou']<0.9:
            shutil.copyfile(path+'test_out/'+key,createPath(path+'WIOU/')+key)
    print('mean iou:',np.mean(IOU),',IOU:',IOU)
    print('WCT-',len(WCT),'-:',WCT)


def sparaseXML(in_file):
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    boxs = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes:# or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text),cls_id]
        boxs.append(b)
    return boxs,w,h,tree


def cretateSubElements(dicts,root):
    for key in dicts:
        elem = Element(str(key))
        elem.text = str(dicts[key])
        root.append(elem)

def saveAnnoInXML(tree,bboxs,path):
    # <object>
    #     <name>in_helmet</name>
    #     <pose>Unspecified</pose>
    #     <truncated>0</truncated>
    #     <difficult>0</difficult>
    #     <bndbox>
    #         <xmin>185</xmin>
    #         <ymin>58</ymin>
    #         <xmax>211</xmax>
    #         <ymax>148</ymax>
    #     </bndbox>
    # </object>
    root = tree.getroot()
    elems = root.findall('object')
    for elem in elems:
        root.remove(elem)
        # pdb.set_trace()
    for box in bboxs:
        # print(box)
        obj = Element('object')
        cretateSubElements({'name':classes[box[-1]],'pose':'Unspecified','truncated':'0','difficult':'0'},obj)
        bndbox = Element('bndbox')
        cretateSubElements({'xmin':box[0],'ymin':box[1],'xmax':box[2],'ymax':box[3]},bndbox)
        obj.append(bndbox)
        root.append(obj)        
    tree.write(path,'utf-8')
    # xmlDoc.write('D:test.xml','utf-8',True)


def getBigAnnoFromSmallPic(xbs,cloumns,raws):
    bbs = []
    for b in xbs:
        bbs.append([b[0]*cloumns,b[1]*raws,b[2]*cloumns,b[3]*raws,b[-1]])
    return bbs

def croped(img,x1,y1,x2,y2):
    return img[y1:y2,x1:x2]

def getHighOverlap(im,ret,bbs,thresh=0.45,outTopThresh = 0.9):
    box2 = ret
    boxs = []
    for box1 in bbs:
        outTop = False
        tb = min(box1[2],box2[2])-max(box1[0],box2[0])
        lr = min(box1[3],box2[3])-max(box1[1],box2[1])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        w1,h1 = box1[2]-box1[0],box1[3]-box1[1]
        sArea = w1*h1 if w1*h1!=0 else 0
        overlap = intersection / sArea
        # print(overlap)
        if box1[1]<ret[1]:
            temp = thresh
            thresh = outTopThresh
            outTop = True
        if overlap<thresh:
            thresh = temp if outTop==True else thresh 
            continue
        # pdb.set_trace()
        # if box1[1]<ret[1]:
            # headX = ret[0]-box1[0] if box1[0]<ret[0] else min(w1,ret[2]-box1[0])
            # headY = ret[1]-box1[1]
            # headRate = headX*headY/sArea
            # if headRate>0.1:
                # continue
        x1 = max(box1[0]-ret[0],0)
        y1 = max(box1[1]-ret[1],0)
        x2 = min(box1[0]-ret[0]+w1,ret[2]-ret[0])
        y2 = min(box1[1]-ret[1]+h1,ret[3]-ret[1])
        boxs.append([x1,y1,x2,y2,box1[-1]])
        thresh = temp if outTop==True else thresh 
    return boxs

def drawBoxs(im,boxs,show=True):
    r = random.randint(0,255)
    g = random.randint(0,255)
    bb = random.randint(0,255)
    for b in boxs:
        # pdb.set_tracqe()
        cv2.rectangle(im,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(bb,g,r),2)
    if show:
        cv2.imshow('im',im)
        cv2.waitKey(10)
        

def getAllCropedPic(imgpath,labelPath,desimgpath,deslabelpath,showOrNot=True):
    files = os.listdir(imgpath)
    fid = open('/mnt/sdb/wsy_dataset/absenceXML.txt','a')
    fid1 = open('/mnt/sdb/wsy_dataset/absenceSmallPic.txt','a')
    for fi in files:
        fname = fi.split('.')[0]
        # if fname!='0789':
        #     continue
        lp = labelpath+fname+'.xml'
        if not os.path.exists(lp):
            fid.write(lp)
            print('absent the xml:',lp)
            continue
        lfi = open(lp,'r')
        xbs,sw,sh,tree = sparaseXML(lfi)  #get the size of small pic
        img = cv2.imread(imgpath+fi)
        (bh,bw,ch) = img.shape
        cloumns = bw//sw
        raws = bh//sh
        if sw>=bw  or sh>=bh:
            fid1.write(lp)
            print('small pic >= big pic:',lp)
            continue
        bboxs = getBigAnnoFromSmallPic(xbs,cloumns,raws)
        # pdb.set_trace()
        # drawBoxs(img,bboxs)
        for st in range(2):
            for i in range(cloumns-st):
                for j in range(raws-st):
                    # if '''_%d_%d_%d'''%(st,i,j) !='_1_0_1':
                    #     continue
                    x1 = st*(sw//2) + j*sw
                    y1 = st*(sh//2) + i*sh
                    x2 = x1 + sw
                    y2 = y1 + sh
                    im = croped(img,x1,y1,x2,y2)
                    # pdb.set_trace()
                    bbs = getHighOverlap(im,[x1,y1,x2,y2],bboxs)
                    if len(bbs)==0:#[1680.0, 321.0, 1764.0, 621.0]
                        continue
                    # drawBoxs(im,bbs,show=showOrNot)
                    cv2.imwrite(desimgpath+fname+'''_%d_%d_%d.png'''%(st,i,j),im)
                    saveAnnoInXML(tree,bbs,desimgpath+fname+'''_%d_%d_%d.xml'''%(st,i,j))
        print(fi,'finished')
    if showOrNot:
        cv2.destroyAllWindows()  

def renameFiles():
    files = os.listdir(files)


def listValPics(picpath,txtpath):
    files = os.listdir(picpath)
    fid = open(txtpath,'w')
    for fi in files:
        fname = picpath+fi
        fid.write(fname+'\n')
        print(fname)
    fid.close()

def getPicFolder(txtpath,spath,despath,fformat):
    fid = open(txtpath)
    lines = fid.readlines()
    nums = []
    for li in lines:
        print(li)
        fi = li.split('/')[-1]
        index = fi.split('_')[0] if '_' in fi else fi.split('.')[0]
        fname = index+fformat
        if index not in nums:
            shutil.copy(spath+fname,despath+fname)
            nums.append(index)
        print(fname)
    fid.close()


def copyFiles(path):
    desJ = createPath('/mnt/sdb/wsy_dataset/darknet/data/helmet/VOC/JPEGImages/')
    desL = createPath('/mnt/sdb/wsy_dataset/darknet/data/helmet/VOC/Annotations/')
    des = createPath('/mnt/sdb/wsy_dataset/croped1/')
    files = os.listdir(path)
    for fi in files:
        ftype = fi.split('.')[1]
        index = int(fi.split('_')[0]) if '_' in fi else int(fi.split('.')[0])
        if index<=720:
            if (ftype=='png' or ftype=='jpg') and not os.path.exists(desJ+fi):      
                shutil.copy(path+fi,desJ+fi)
            elif ftype=='xml' and not os.path.exists(desL+fi):
                shutil.copy(path+fi,desL+fi)
            # if not os.path.exists(des+fi):
            #     shutil.copy(path+fi,des+fi)
            print('c:',fi)
        # img = cv2.imread(path+fi)
        # (ph,pw,ch) = img.shape #the size of mat,(raws,cloumns,ch)
        # print('pw',pw,'ph',ph)
        # # break
        # if pw!=1920 or ph!=1080:
        #     shutil.move(path+fi,des+fi)






if __name__ == '__main__':
    # path = '/home/lab601/project/data/Test_fix/'
    # fid = open('data/gp/voc/VOCdevkit/test.txt','w')
    # files = os.listdir(path)
    # for f in files:
    #     line = path+f
    #     fid.write(line+'\n')
    # year = 2007
    # image_set = 'trainval.txt'
    # image_ids = open('data/wp/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    # for image_id in image_ids:
    #     # pdb.set_trace()
    #     convert_annotation('2007',image_id.strip())

    # --------------------wsy--------------------------
    # loadJson()
    # persons={'ZhangSan':'male', 'LiSi':'male', 'WangHong':'female'} #找出所有男性 
    # males = filter(lambda x:'male'== x[1], persons.items())
    # pdb.set_trace()
    # for (key,value) in males:
    #     print('%s : %s' % (key,value))

    # path = 'data/gp/voc/VOCdevkit/'
    # cfile = path+'Test_result_all_res50_config_v1_6fpn_libra_v1.csv'
    # file = path+'test_result.csv'
    # cmpCSV(cfile,file)

    imgpath = '/mnt/sdb/wsy_dataset/bigPic/allpics/'
    labelpath = '/mnt/sdb/wsy_dataset/helmet/ImageSets/'
    desimgpath = createPath('/mnt/sdb/wsy_dataset/croped1/')
    deslabelpath = '/mnt/sdb/wsy_dataset/helmet/JPEGImages/'

    imgdespath = createPath('/mnt/sdb/wsy_dataset/mAP-master/input/images-optional/')
    gtdespath = createPath('/mnt/sdb/wsy_dataset/mAP-master/input/ground-truth/')
    txtpath = 'data/helmet/VOC/bigPicVal.txt'
    resizeXMLPath = '/mnt/sdb/wsy_dataset/helmet/Annotations/'

    getAllCropedPic(imgpath,labelpath,desimgpath,deslabelpath)
    # test(imgpath)
    # test(imgpath)
    # fformat = '.xml'
    # getPicFolder(txtpath,resizeXMLPath,gtdespath,fformat)
    # listValPics(imgdespath,txtpath)
