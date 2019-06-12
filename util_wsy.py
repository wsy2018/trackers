
import os

def saveInFile(frame,boxes,path,filename,method):
    if boxes is None or len(boxes)<1:
        return
    path = createPath(path)
    f1 = open(path+filename,method)
    print(boxes)
    for d in boxes:
        x = d[0]
        y = d[1]
        w = d[2]-d[0]
        h = d[3]-d[1]
        score = d[4]
        line = str(frame)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+' '+str(score)+'\n'
        f1.write(line)
    f1.close()

def saveNameInFile(frame,boxes,path,filename,method):
    if boxes is None or len(boxes)<1:
        return
    path = createPath(path)
    f1 = open(path+filename,method)
    print(boxes)
    # pdb.set_trace()
    for d in boxes:
        x1 = d[0]
        y1 = d[1]
        x2 = d[2]
        y2 = d[3]
        score = d[4]
        line = 'rf_result_frame_'+str(frame)+'_'+str(x1)+'_'+str(y1)+'_'+str(x2)+'_'+str(y2)+':'+str(score)+'\n'
        f1.write(line)
    f1.close()

def readGtTxt(path,filename,method):
    gt = []
    f1 = open(path+filename,method)
    lines = f1.readlines()
    for li in lines:
        d = li.split(',')
        x1 = int(float(d[0]))
        y1 = int(float(d[1]))
        x2 = int(float(d[4]))
        y2 = int(float(d[5]))               
        gt.append([x1,y1,x2,y2])
    f1.close()
    return gt

def readVocGt(path,filename,method='r'):
    gt = []
    fid = open(path+filename,method)
    lines = fid.readlines()
    for li in lines:
        d = li.split(' ')
        x1 = int(float(d[1]))
        y1 = int(float(d[2]))
        x2 = int(float(d[3]))
        y2 = int(float(d[4]))               
        gt.append([d[0],x1,y1,x2,y2])
    fid.close()
    return gt

def CountIOU(prebox,gt):
	ious = []
	for pre in prebox:
		for g in gt:
			if g[0]==pre[0]:
				ious.append(iou(pre[1:],g[1:]))
	return ious

def iouWH(box1,box2):
    tb = min(box1[0]+box1[2],box2[0]+box2[2])-max(box1[0]-box1[2],box2[0]-box2[2])
    lr = min(box1[1]+box1[3],box2[1]+box2[3])-max(box1[1]-box1[3],box2[1]-box2[3])
    if tb < 0 or lr < 0 : intersection = 0
    else : intersection =  tb*lr
    if (box1[2]*box1[3] + box2[2]*box2[3] - intersection) ==0 :
        return 0
    else :    
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

def iouLT(box1,box2):
    tb = min(box1[2],box2[2])-max(box1[0],box2[0])
    lr = min(box1[3],box2[3])-max(box1[1],box2[1])
    w1,h1 = box1[2]-box1[0],box1[3]-box1[1]
    w2,h2 = box2[2]-box2[0],box2[3]-box2[1]

    if tb < 0 or lr < 0 : intersection = 0
    else : intersection =  tb*lr
    
    unoi = w1*h1 + w2*h2 - intersection
    if unoi ==0 :
        return 0
    else :    
        return intersection / unoi

def detect_video(yolo, video_path):
    vid = cv2.VideoCapture(video_path)
    # pdb.set_trace()
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    itr = 0
    frames = []
    while True:
        itr = itr + 1
        return_value, frame = vid.read()
        pdb.set_trace()
        if frame is None:
            break
        image = Image.fromarray(frame)
        image_data = np.array(image, dtype='float32')
        #cv2.imwrite(createPath('./rabbit/org_pic/')+'frame_'+str(itr)+'.png',image_data)
        image,boxes = yolo.detect_Frame_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        #scipy.misc.imsave(img_save_path+str(itr)+'_'+str(curr_fps)+'.png',result);
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        frames.append(result)
        #saveInFile(itr,boxes,'./','yolo_v3_dv_7.txt','a')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    transfImg2Video(video_path,frames)
    # yolo.close_session()

def transfImg2Video(video_path,frames):
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    spstrs = video_path.split('/')
    vtype = spstrs[-2]+'/'
    sequence_name = spstrs[-1]
    video_name = 'result_'+sequence_name

    ht,wid,_=frames[0].shape
    video_path = os.path.join(createPath('./result/'+vtype), video_name)
    video = cv2.VideoWriter(video_path, fourcc, 20, (wid, ht))
    framepath = createPath('./result/'+vtype+sequence_name+'/')
    for i,frame in enumerate(frames):
        video.write(frame1)
        # cv2.imwrite(framepath+'frame_'+str(i)+'.png',frame)
    video.release()

def createPath(path):
  isexist = os.path.exists(path)
  if not isexist:
      os.makedirs(path)
  return path

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.save('result.png')
            r_image.show()
    yolo.close_session()

def detect_img1(yolo,path): 
    imgs = [os.path.join(path,name) for name in os.listdir(path)]
    imgs = sorted(imgs)
    for img in imgs:
        # img = input('Input image filename:')
        try:
            image = Image.open(img)
            imgdata = np.array(image)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.save(createPath(path+'out/')+img.split('/')[-1])
            imgdata = np.array(r_image)
            cv2.imshow("result", imgdata)
            cv2.waitKey(5)
            # r_image.show()
    cv2.destroyAllWindows()
    yolo.close_session()

def transfImg2Video1(imgpath,video_path,frate):
    # video_path = './'
    # pdb.set_trace()
    frames = []
    path  = imgpath
    imgpath = [os.path.join(path,name) for name in os.listdir(path)]
    imgpath=sorted(imgpath)
    for i,ipt in enumerate(imgpath):
        frame = cv2.imread(ipt)
        frames.append(frame)
        if i==0:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            ht,wid,_=frames[0].shape
            video = cv2.VideoWriter(video_path,fourcc,frate, (wid, ht))
        video.write(frame)
        # cv2.imwrite(framepath+'frame_'+str(i)+'.png',frame)
    video.release()


def transfImg2GtVideo1(imgpath,video_path,frate):
    # video_path = './'
    path  = imgpath
    imgpath = [os.path.join(path,name) for name in os.listdir(path)]
    imgpath=sorted(imgpath)
    gt = readGtTxt(path,'groundtruth.txt','r')
    # pdb.set_trace()
    for i,ipt in enumerate(imgpath):
        if i>len(gt):
            break
        try:
            image = Image.open(ipt)
        except:
            print('Open Error! Try again!')
            continue
        else:
            # pdb.set_trace()
            draw = ImageDraw.Draw(image)
            draw.rectangle([gt[i][0],gt[i][1],gt[i][2],gt[i][3]],outline=(0,0,255))
            frame = np.array(image)
            if i==0 :
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                ht,wid,_=frame.shape
                video = cv2.VideoWriter(video_path,fourcc,frate, (wid, ht))
            if frame is not  None:
                frame1 =  cv2.resize(frame, (wid,ht), interpolation=cv2.INTER_AREA) 
                video.write(frame1)
                cv2.imwrite(createPath(path+'gt_box_pic/')+ipt.split('/')[-1],frame)
                print(path+'/gt_box_pic/frame_'+str(i)+'.png')
    video.release()



def detect_file_imgs(yolo,fpath,fname,method='r'):
    f1 = open(fpath+fname,method)
    lines = f1.readlines()
    filenames = []
    ious = []
    pdb.set_trace()
    for img in lines:
        # img = input('Input image filename:')
        try:
            image = Image.open(img.strip('\n'))
            imgdata = np.array(image)
            # pdb.set_trace()
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image,prebox = yolo.detect_Frame_image(image)
            filenames = img.strip('.jpg').split('/')[-1]
            gt = readVocGt('',filename+'.txt')
            ious.extend(CountIOU(prebox,gt))
            pdb.set_trace()
            r_image.save(createPath('./wp_test_out/')+filenames+'.jpg')
            imgdata = np.array(r_image)
            # frames.append(imgdata)
            cv2.imshow("result", imgdata)
            cv2.waitKey(3)
            # r_image.show()
    cv2.destroyAllWindows()
    yolo.close_session()
