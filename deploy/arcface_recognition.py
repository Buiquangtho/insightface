# -*- coding: UTF-8 -*-  
"""
Created on Tue Jul  9 18:34:40 2019

@author: linhenrycw
#version 2: display rectangle and text by PIL for 中文 label
"""

import face_model
import argparse
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw, ImageFont #for 顯示圖片 & 中文字label
import glob
import os
import time
from timeit import default_timer as timer


registed_folder = 'registed_img_mobile'
#fontText = ImageFont.truetype("font/simsun.ttc", 23, encoding="utf-8")

def face_registed_projector(model):
    #run when have not registed to npy file
    img_list_jpg = glob.glob(os.path.join(cwd,'img_for_registed','*.jpg'))
    img_list_jpeg = glob.glob(os.path.join(cwd,'img_for_registed','*.jpeg'))
    img_list_png = glob.glob(os.path.join(cwd,'img_for_registed','*.png'))
    img_list = img_list_jpg + img_list_jpeg + img_list_png
    fact_cnt = 0
    for file in img_list:
        #img = cv2.imread(file)
        img=cv2.imdecode(np.fromfile(file,dtype=np.uint8),-1)
        if '.jp' in file:
            file_name = file.split('\\')[-1].split('.jp')[0]
        elif '.png' in file:
            file_name = file.split('\\')[-1].split('.png')[0]
        elif '.JPG' in file:
            file_name = file.split('\\')[-1].split('.JPG')[0]
        print('registering for %s'%file_name)
        try:
            faces,points,bbox = model.get_input(img)
        except:
            print('fail to read %s, which will be ommitted'%file_name)		
        
        for i in range(faces.shape[0]):
            face = faces[i]
            f1 = model.get_feature(face)
            #print('feature shape:')
            #print(f1.shape)
        
            margin = 44
            x1 = int(np.maximum(np.floor(bbox[i][0]-margin/2), 0) )
            y1 = int(np.maximum(np.floor(bbox[i][1]-margin/2), 0) )
            x2 = int(np.minimum(np.floor(bbox[i][2]+margin/2), img.shape[1]) )
            y2 = int(np.minimum(np.floor(bbox[i][3]+margin/2), img.shape[0]) )

            	
            if i>=1:
                npy_name = file_name+'_'+str(i)+'.npy'
                jpg_name = file_name+'_'+str(i)+'.jpg'
            else:
                npy_name = file_name+'.npy'
                jpg_name = file_name+'.jpg'
            np.save(os.path.join(cwd,registed_folder,npy_name),f1)
            #cv2.imwrite(os.path.join(cwd,registed_folder,jpg_name), img[y1:y2, x1:x2])
            cv2.imencode('.jpg', img[y1:y2, x1:x2])[1].tofile(os.path.join(cwd,registed_folder,jpg_name))
            fact_cnt+=1
    print('成功註冊 %d 張照片，共 %d 張臉!'%(len(img_list),fact_cnt))


def registed_face_loader():
    ## load registed npy ##
    registed_npy_list = glob.glob(os.path.join(cwd,registed_folder,'*.npy'))
    registed_feature = []
    cat = []
    for npy in registed_npy_list:
        f1 = np.load(npy)
        cat.append(npy.split('\\')[-1].split('.npy')[0])
        registed_feature.append(f1)
    if registed_npy_list==[]:
        print('there is no .npy file in registed face folder')
    else:
        print('load registed %d faces with %d dimensions'%(len(registed_feature),registed_feature[0].shape[0]))
    print('registed names:')
    print(cat)
    return registed_feature,cat



def face_comparison(img,registed_feature,cat,model,threshold = 0.45):

    faces,points,bbox = model.get_input(img)
    
    #cv2 format to PIL format for 中文字label
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #cv2 to PIL
	
    all_cat,all_sim = [],[]
    if (faces==[] or points==[] or bbox==[]):
      img = img
    
    else:
        text_size = np.floor(2e-2 * img_PIL.size[1]).astype('int32')
        fontText = ImageFont.truetype("font/simsun.ttc", text_size, encoding="utf-8")
        thickness = (img_PIL.size[0] + img_PIL.size[1]) // 600
        for i in range(faces.shape[0]):
            
            face = faces[i]
            f2 = model.get_feature(face)
            
            #gender, age = model.get_ga(face)
	      
            sim_record = []
            for j in range(len(registed_feature)):
                sim_record.append(np.dot(registed_feature[j], f2.T))
            
            most_sim_ind = sim_record.index(max(sim_record))
                       
            margin = 44
            x1 = int(np.maximum(np.floor(bbox[i][0]-margin/2), 0) )
            y1 = int(np.maximum(np.floor(bbox[i][1]-margin/2), 0) )
            x2 = int(np.minimum(np.floor(bbox[i][2]+margin/2), img.shape[1]) )
            y2 = int(np.minimum(np.floor(bbox[i][3]+margin/2), img.shape[0]) )
            
            draw = ImageDraw.Draw(img_PIL)
            if sim_record[most_sim_ind]>=threshold:
                text = cat[most_sim_ind]+','+str(np.round(sim_record[most_sim_ind],3))
                
                label_size = draw.textsize(text, fontText)
                if y1 - label_size[1] >= 0:
                    text_origin = np.array([x1, y1 - label_size[1]])
                else:
                    text_origin = np.array([x1, y1 + 1])
                
                for i in range(thickness):
                    draw.rectangle([x1+i,y2+i,x2-i,y1-i],outline="red")#畫框
                    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],outline="red",fill="red")#字體background的框
                    draw.text(text_origin, text, (255, 255, 255), font=fontText)
	    	  
            else:
                for i in range(thickness):
                    draw.rectangle([x1+i,y2+i,x2-i,y1-i],outline="green" )#畫框
            
            del draw
            
            all_cat.append(cat)
            all_sim.append(sim_record)
        
        
    return all_cat,all_sim,img_PIL


def face_comparison_video(registed_feature,cat,model,input_path,output_path,threshold = 0.45):
    if input_path=='':
	    vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(input_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        
        all_cat,all_sim,image = face_comparison(frame,registed_feature,cat,model,threshold = 0.45)
        img_OpenCV = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) #轉回cv2 format 否則RGB亂掉
        result = np.asarray(img_OpenCV)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face embedding and comparison')
    parser.add_argument('--image-size', default='112,112', help='') #follow sphere face & cosin face format
    parser.add_argument('--model', default='./models/model-mobileface/model,0', help='path to load model.')
    #parser.add_argument('--ga-model', default='./models/gamodel-r50/model,0', help='path to load model.')
    parser.add_argument('--gpu', default=-1, type=int, help='gpu id,(-1) for CPU')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    parser.add_argument('--image_mode', default=False, action="store_true",help='Image detection mode')
    parser.add_argument('--regist', default=False, action="store_true",help='to regist face or to compare face')
    args = parser.parse_args()
    
    while True:
        regist_mode_input = input('是否註冊臉模式?(0 or 1):')
        if regist_mode_input in ["0","1"]:
            args.regist = int(regist_mode_input)
            if args.regist==0:
                while True:
                    image_mode_input = input('照片模式?(0 for video, 1 for 照片):')
                    if image_mode_input in ["0","1"]:
                        args.image_mode = int(image_mode_input)
                        if args.image_mode ==0:
                            video_input_path = input('input video path or just press enter to use web camera:')
                            if video_input_path!='':
                                video_output_path = input('video detect result output path:')
                            else:
                                video_output_path = ''
                        break
                    else:
                        print('please follow the input rule!!')
                break
            else:
                break
        else:
            print('please follow the input rule!!')
    
    model = face_model.FaceModel(args)
    cwd = os.getcwd()

    if args.regist:
        face_registed_projector(model)
        
    else:
        registed_feature,cat = registed_face_loader()
        if args.image_mode:
            while True:
                img_path = input('input the image path:')
                try:
                    #img = cv2.imread(img_path)
                    img=cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
                    start_time = time.time()
                    all_cat,all_sim,img_result = face_comparison(img,registed_feature,cat,model)
                    elapsed_time = time.time() - start_time
                    print('**cost %s time for compare %d face with %d face in database'%(elapsed_time,len(all_cat),len(cat)))
                
                    img_result.show()
                    img_result.save('out.jpg')
                except:
                    print('cannot read image in your input path')
                    continue
				
                #for print result
                for i in range(len(all_cat)):
                    cat = all_cat[i]
                    sim = all_sim[i]
                    tmp = []
                    for c in range(len(cat)): #combine to tuple
                       tmp.append(tuple([cat[c],sim[c]]))
                    
                    result_sorted = sorted(tmp, key=lambda s: s[1], reverse=True)#sort by sim
                    print('**top 5 similar face in db are:')
                    print(result_sorted[:5])

        else:
            face_comparison_video(registed_feature,cat,model,video_input_path,video_output_path,threshold = 0.45)



