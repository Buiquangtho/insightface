import flask
from flask import render_template,request,redirect,url_for

import face_model
import argparse
import cv2
import numpy as np
import os
import logging
import datetime
import json
import face_recognition
import shutil
import glob
from flask_uploads import UploadSet, IMAGES,configure_uploads,patch_request_class

cwd = os.getcwd()
app = flask.Flask(__name__)
app.config["IMAGE_UPLOADS"] = os.path.join(cwd,'uploads')
app.config["IMAGE_REGISTER"] = os.path.join(cwd,'img_for_registed')
app.config["DEBUG"] = True

normalLogger = logging.getLogger('normalLogger')

@app.route('/', methods=['GET'])
def home():
    return "<h1>Hello Flask!</h1>"


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    global image_name
    global file_name_list
    global face_cnt
    if request.method == "POST":

        if request.files and request.form['submit_button'] == 'recognition':
            image = request.files["image"]
            
            image_name = image.filename
            hist_img_list = os.listdir(app.config["IMAGE_UPLOADS"])
            
            ind = 0
            while image_name in hist_img_list:
                tmp_name = image_name.split('--')[-1] # rename the repeat file name
                image_name = str(ind)+'--'+tmp_name
                ind += 1
            img_path = os.path.join(app.config["IMAGE_UPLOADS"], image_name)
            
            normalLogger.debug('receive file: '+image_name)
            image.save(img_path)
            
            normalLogger.debug('start to recognition...')
            img_result,result_dict = recognition(img_path,registed_feature,cat,model)  
            
            return redirect(url_for("result",messages=json.dumps(result_dict,cls=MyEncoder,ensure_ascii=False)))
        
        
        elif request.files and request.form['submit_button'] == 'register':
            image = request.files["image"]
            image_name = image.filename
            img_path = os.path.join(app.config["IMAGE_REGISTER"] , image_name)
            image.save(img_path)
            normalLogger.debug('receive file %s and start to do face detection...: '%image_name)
            
            file_name_list = face_recognition.face_registed_projector(model,registed_folder)
            face_cnt = len(file_name_list)
            for jpg_name in file_name_list:
                shutil.copy(os.path.join(cwd,registed_folder,jpg_name),os.path.join(cwd,'static','register_face',jpg_name))
            
            shutil.move(img_path,os.path.join(cwd,'register_upload_hist',image_name))
            
            return redirect(url_for("register_face"))
        
        
        elif request.form['submit_button'] == 'manage register list':
            return redirect(url_for("manage"))


        	
    tmp = glob.glob(os.path.join(cwd,registed_folder,'*.jpg'))
    registered_face_list = [f.split('\\')[-1] for f in tmp]
    	
    return render_template("upload_image.html",registered_face_list = registered_face_list)


@app.route('/result')
def result():
    full_filename = '/static/result/'+image_name
    return render_template("result.html", user_image=full_filename,messages=request.args.get('messages'))


@app.route('/register_face', methods=["GET", "POST"])
def register_face(): 
    global oper_jpg_name
    global registed_feature
    global cat
    
    if face_cnt>0:        
        if request.method == 'POST':
            ori_npy_name = oper_jpg_name.split('.jpg')[0] + '.npy' 
            input_name = request.values['face_name']
            if input_name != '':
                normalLogger.debug('rename the file from %s to %s' %(oper_jpg_name.split('.jpg')[0],input_name))
                
                new_jpg_name = input_name + '.jpg'
                new_npy_name = input_name + '.npy'
                os.rename(os.path.join(cwd,registed_folder,oper_jpg_name), os.path.join(cwd,registed_folder,new_jpg_name))
                os.rename(os.path.join(cwd,registed_folder,ori_npy_name), os.path.join(cwd,registed_folder,new_npy_name))
            else:
                normalLogger.debug('remove not face file %s' %oper_jpg_name.split('.jpg')[0])
                os.remove(os.path.join(cwd,registed_folder,oper_jpg_name))
                os.remove(os.path.join(cwd,registed_folder,os.path.join(cwd,registed_folder,ori_npy_name) ))
        
        if len(file_name_list)>0:
            show_jpg_name = file_name_list[0]
            full_filename = '/static/register_face/'+show_jpg_name
            oper_jpg_name = show_jpg_name 
            file_name_list.remove(show_jpg_name)		
            return render_template("register.html", face=full_filename)
        else:
            registed_feature,cat = face_recognition.registed_face_loader(registed_folder) #after register new face, reload the registered feature
            return render_template("register_no_face.html")
    
    else: #no face is detected
        return render_template("register_no_face.html")



###############

@app.route('/manage')
def manage():
    files_list = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
    return render_template('manage.html', files_list=files_list)


@app.route('/open/<filename>')
def open_file(filename):
    file_url = photos.url(filename)
    return render_template('browser.html', file_url=file_url)


@app.route('/delete/<filename>')
def delete_file(filename):
    global registed_feature
    global cat
    file_path = photos.path(filename)
    os.remove(file_path)
    registed_feature,cat = face_recognition.registed_face_loader(registed_folder) #after delete face, reload the registered feature
    return redirect(url_for('manage'))

##################




class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)	


def SetupLogger(loggerName, filename):
    path = os.path.join(cwd,'logs')
    if not os.path.exists(path):
        os.makedirs(path)

    logger = logging.getLogger(loggerName)

    logfilename = datetime.datetime.now().strftime(filename)
    logfilename = os.path.join(path, logfilename)
    
    logformatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    fileHandler = logging.FileHandler(logfilename, 'a', 'utf-8')
    fileHandler.setFormatter(logformatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(logformatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)


def recognition(img_path,registed_feature,cat,model):
    img=cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
    normalLogger.debug('successfully read uploded image. start to compare face...')
    
    all_cat,all_sim,img_result,result_dict = face_recognition.face_comparison(img,registed_feature,cat,model)
    normalLogger.debug('finish face comparion, save result image...')
    img_result.save(os.path.join(cwd,'static','result',image_name))
    
    '''
    for i in range(len(all_cat)):
        cat = all_cat[i]
        sim = all_sim[i]
        tmp = []
        for c in range(len(cat)): #combine to tuple
           tmp.append(tuple([cat[c],sim[c]]))
        
        result_sorted = sorted(tmp, key=lambda s: s[1], reverse=True)#sort by sim
    '''
    return img_result,result_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face embedding and comparison')
    parser.add_argument('--image-size', default='112,112', help='') #follow sphere face & cosin face format
    parser.add_argument('--model', default='./models/model-mobileface/model,0', help='path to load model.')
    parser.add_argument('--gpu', default=-1, type=int, help='gpu id,(-1) for CPU')
    parser.add_argument('--cam', default=0, type=int, help='which cam used')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    parser.add_argument('--image_mode', default=False, action="store_true",help='Image detection mode')
    parser.add_argument('--regist', default=False, action="store_true",help='to regist face or to compare face')
    args = parser.parse_args()
    SetupLogger('normalLogger', "%Y-%m-%d.log")
    
    model = face_model.FaceModel(args)
    if 'mobileface' in args.model:
        registed_folder = 'registed_img_mobile'
    else:
        registed_folder = 'registed_img_r100'
    app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(cwd,registed_folder)
    photos = UploadSet('photos', IMAGES)
    configure_uploads(app, photos)
    patch_request_class(app)  # set maximum file size, default is 16MB
    
    
    registed_feature,cat = face_recognition.registed_face_loader(registed_folder)
    
    app.run(threaded=False,processes=1)




