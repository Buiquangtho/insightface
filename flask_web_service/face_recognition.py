from PIL import Image,ImageDraw, ImageFont #for 顯示圖片 & 中文字label
import glob
import os
import logging
import datetime
import numpy as np
import cv2


cwd = os.getcwd()
normalLogger = logging.getLogger('normalLogger')
	
def face_registed_projector(model,registed_folder):
    #run when have not registed to npy file
    img_list_jpg = glob.glob(os.path.join(cwd,'img_for_registed','*.jpg'))
    img_list_jpeg = glob.glob(os.path.join(cwd,'img_for_registed','*.jpeg'))
    img_list_png = glob.glob(os.path.join(cwd,'img_for_registed','*.png'))
    img_list = img_list_jpg + img_list_jpeg + img_list_png
    fact_cnt = 0
    jpg_name_list = []
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
        normalLogger.debug('registering for %s'%file_name)			
        try:
            faces,points,bbox = model.get_input(img)
        except:
            print('fail to read %s, which will be ommitted'%file_name)
            normalLogger.debug('fail to read %s, which will be ommitted'%file_name)			
        
        for i in range(len(faces)):
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
            jpg_name_list.append(jpg_name)
    print('成功註冊 %d 張照片，共 %d 張臉!'%(len(img_list),fact_cnt))
    normalLogger.debug('成功註冊 %d 張照片，共 %d 張臉!'%(len(img_list),fact_cnt))
    
    
    return jpg_name_list
    

def registed_face_loader(registed_folder):
    ## load registed npy ##
    registed_npy_list = glob.glob(os.path.join(cwd,registed_folder,'*.npy'))
    registed_feature = []
    cat = []
    for npy in registed_npy_list:
        f1 = np.load(npy)
        cat.append(npy.split('\\')[-1].split('.npy')[0])
        registed_feature.append(f1)
    if registed_npy_list==[]:
        normalLogger.debug('there is no .npy file in registed face folder')
    else:
        normalLogger.debug('load registed %d faces with %d dimensions'%(len(registed_feature),registed_feature[0].shape[0]))
    normalLogger.debug('registed names: '+str(cat))
    return registed_feature,cat



def face_comparison(img,registed_feature,cat,model,threshold = 0.45):
    faces,points,bbox = model.get_input(img)
    normalLogger.debug('finish face detection, start to embed face')
    #cv2 format to PIL format for 中文字label
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #cv2 to PIL
	
    all_cat,all_sim = [],[]
    if (faces==[] or points==[] or bbox==[]):
        img = img  
        result_dict = {"most_sim_face":None,"similarity":None,"bbox":None}
    else:
        text_size = np.floor(3e-2 * img_PIL.size[1]).astype('int32')
        fontText = ImageFont.truetype("font/simsun.ttc", text_size, encoding="utf-8")
        thickness = (img_PIL.size[0] + img_PIL.size[1]) // 600


        bbox_sent = []
        most_sim_face = []
        similarity = []
        normalLogger.debug('finish face embedding! start to compare face...')
        for i in range(faces.shape[0]):
            
            face = faces[i]
            f2 = model.get_feature(face)
            
            
            sim_record = np.matmul(registed_feature,f2.T)
            most_sim_ind = np.argmax(sim_record)
             			
            margin = 30
            x1 = int(np.maximum(np.floor(bbox[i][0]-margin/2), 0) )
            y1 = int(np.maximum(np.floor(bbox[i][1]-margin/2), 0) )
            x2 = int(np.minimum(np.floor(bbox[i][2]+margin/2), img.shape[1]) )
            y2 = int(np.minimum(np.floor(bbox[i][3]+margin/2), img.shape[0]) )
            
            sub_bbox_str = "%d,%d,%d,%d"%(x1,y1,x2,y2)
            bbox_sent.append(sub_bbox_str)

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
			
            most_sim_face.append(cat[most_sim_ind])
            similarity.append(sim_record[most_sim_ind])
        result_dict = {"most_sim_face":most_sim_face,"similarity":similarity,"bbox":bbox_sent}
        
        normalLogger.debug(str(result_dict))

    return all_cat,all_sim,img_PIL,result_dict
