# -*- coding: UTF-8 -*-  
"""
Created on Tue Jul  9 18:34:40 2019

@author: linhenrycw
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


registed_folder = 'registed_img_r100'


def face_registed_projector(model):
    #run when have not registed to npy file
    img_list_jpg = glob.glob(os.path.join(cwd,'img_for_registed','*.jpg'))
    img_list_jpeg = glob.glob(os.path.join(cwd,'img_for_registed','*.jpeg'))
    img_list_png = glob.glob(os.path.join(cwd,'img_for_registed','*.png'))
    img_list = img_list_jpg + img_list_jpeg + img_list_png
    
    for file in img_list:
        img = cv2.imread(file)
        if '.jp' in file:
            file_name = file.split('\\')[-1].split('.jp')[0]
        elif '.png' in file:
            file_name = file.split('\\')[-1].split('.png')[0]
        
        faces,points,bbox = model.get_input(img)
        for i in range(faces.shape[0]):
            face = faces[i]
            f1 = model.get_feature(face)
            print('feature shape:')
            print(f1.shape)
        
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
            cv2.imwrite(os.path.join(cwd,registed_folder,jpg_name), img[y1:y2, x1:x2])



def registed_face_loader():
    ## load registed npy ##
    registed_npy_list = glob.glob(os.path.join(cwd,registed_folder,'*.npy'))
    registed_feature = []
    cat = []
    for npy in registed_npy_list:
        f1 = np.load(npy)
        cat.append(npy.split('\\')[-1].split('.npy')[0])
        registed_feature.append(f1)

    print('load registed %d faces with %d dimensions'%(len(registed_feature),registed_feature[0].shape[0]))
    print('registed names:')
    print(cat)
    return registed_feature,cat



def face_comparison(img,registed_feature,cat,model,threshold = 0.3):

    faces,points,bbox = model.get_input(img)

    all_cat,all_sim = [],[]
    if (faces==[] or points==[] or bbox==[]):
      img = img
    
    else:
      for i in range(faces.shape[0]):
          
          face = faces[i]
          f2 = model.get_feature(face)
          sim_record = []
          for j in range(len(registed_feature)):
              sim_record.append(np.dot(registed_feature[j], f2.T))
          
          most_sim_ind = sim_record.index(max(sim_record))
                  
          
          margin = 44
          x1 = int(np.maximum(np.floor(bbox[i][0]-margin/2), 0) )
          y1 = int(np.maximum(np.floor(bbox[i][1]-margin/2), 0) )
          x2 = int(np.minimum(np.floor(bbox[i][2]+margin/2), img.shape[1]) )
          y2 = int(np.minimum(np.floor(bbox[i][3]+margin/2), img.shape[0]) )
  
          cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
          if sim_record[most_sim_ind]>=0.3:
              text = cat[most_sim_ind]+','+str(np.round(sim_record[most_sim_ind],3))
          else:
              text = '???'
          cv2.putText(img, text, (x1,y1-5), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255))
          
          all_cat.append(cat)
          all_sim.append(sim_record)
          #plt.subplot(faces.shape[0]+1,1,i+1)
          #plt.imshow(np.transpose(face,(1,2,0)),aspect='equal')
          #plt.title('most similar face:%s, similarity is:%f'%(cat[most_sim_ind],sim_record[most_sim_ind]))
      #plt.savefig('1.png', bbox_inches = 'tight')
      #plt.show()
    
    	
    return all_cat,all_sim,img


def face_comparison_video(registed_feature,cat,model,threshold = 0.3, output_path=""):
    vid = cv2.VideoCapture(0)
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
        
        all_cat,all_sim,image = face_comparison(frame,registed_feature,cat,model,threshold = 0.3)
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
    parser.add_argument('--model', default='../models/model-r100-ii/model,0', help='path to load model.')
    parser.add_argument('--ga-model', default='../models/gamodel-r50/model,0', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id,(-1) for CPU')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    parser.add_argument('--image_mode', default=False, action="store_true",help='Image detection mode')
    parser.add_argument('--regist', default=False, action="store_true",help='to regist face or to compare face')
    args = parser.parse_args()
    
    model = face_model.FaceModel(args)
    cwd = os.getcwd()

    if args.regist:
        face_registed_projector(model)
        
    else:
        registed_feature,cat = registed_face_loader()
        if args.image_mode:
            while True:
                img_path = input('input the image path:')
                img = cv2.imread(img_path)
                start_time = time.time()
                all_cat,all_sim,img_result = face_comparison(img,registed_feature,cat,model)
                elapsed_time = time.time() - start_time
                print('**cost %s time for compare %d face with %d face in database'%(elapsed_time,len(all_cat),len(cat)))
                
                img2 = img_result[: , : , : : -1]
                img2 = Image.fromarray(img2, 'RGB')
                img2.show()
                img2.save('out.jpg')
				
				
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
            face_comparison_video(registed_feature,cat,model,threshold = 0.3, output_path="")





############# test #####
'''
model = face_model.FaceModel(args)
img = cv2.imread('woman.png')
all_face,all_points = model.get_input(img)

for i in range(all_face.shape[0]):
    face = all_face[i]
    f1 = model.get_feature(face)
    gender, age = model.get_ga(face)
    print('sex:%d'%gender)
    print('age:%d'%age)
    plt.imshow(np.transpose(face,(1,2,0)))
    plt.title('sex:%d, age:%d'%(gender,age))
    plt.show()



img = cv2.imread('two_people.jpg')
all_face,all_points = model.get_input(img)

plt.figure(figsize=(12, 12))	
for i in range(all_face.shape[0]):
    face = all_face[i]
    f2 = model.get_feature(face)
    
    sim = np.dot(f1, f2.T)
    print('%d face -- similarity:%f'%(i,sim))
    
    plt.subplot(all_face.shape[0]+1,1,i+1)
    plt.imshow(np.transpose(face,(1,2,0)),aspect='equal')
    plt.title('similarity:%f'%sim)
plt.savefig('1.png', bbox_inches = 'tight')
plt.show()
#########
'''

####################忘記這邊幹嘛的##########################################################

'''
###
f1 = model.get_feature(img)
#print(f1[0:10])
gender, age = model.get_ga(img)
print('sex:%d'%gender)
print('age:%d'%age)
print(img.shape)
plt.imshow(np.transpose(img,(1,2,0)))
plt.show()
#sys.exit(0)


#=========== compare figure =====#
img = cv2.imread('two_people.jpg')
#cv2.imshow('show',img)
#cv2.waitKey(2000)
img = model.get_input(img)
f2 = model.get_feature(img)
dist = np.sum(np.square(f1-f2))
print('dist:%f'%dist)
sim = np.dot(f1, f2.T)
print('similarity:%f'%sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
'''
