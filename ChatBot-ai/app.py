
import os
import math
import datetime
import re
import io
import sys
sys.path
import zlib
from werkzeug.utils import secure_filename
from flask import Response
from cs50 import SQL
from flask import Flask, flash, jsonify, redirect, render_template, request, session ,url_for
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
import face_recognition
from PIL import Image
from base64 import b64encode, b64decode
from tqdm import tqdm
import pandas as pd
import numpy as np
from helpers import apology, login_required
import tensorflow 
from tensorflow import keras
import time
from threading import Thread
import cv2
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

import torch
global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
thres=0
face=0
switch=1
rec=0

try:
    os.mkdir('./shots')
except OSError as error:
    pass

net = cv2.dnn.readNetFromCaffe('savedmodel/deploy.prototxt.txt', 'savedmodel/res10_300x300_ssd_iter_140000.caffemodel')

bert_model_name="uncased_L-12_H-768_A-12"
bert_ckpt_dir=os.path.join("C:/PythonProj/ChatbotB",bert_model_name)

bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import bert
from bert import BertModelLayer
from transformers import BertForQuestionAnswering
from tensorflow.keras.models import load_model
model = load_model('C:/PythonProj/ChatbotB/colab.h5', custom_objects={"BertModelLayer": bert.model.BertModelLayer})
import json
import random

print(tensorflow.__version__)
Qnamodel = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
from transformers import BertTokenizer

tokenizer_Qna = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    
    
    input_ids = tokenizer_Qna.encode(question, answer_text)

    
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    
    
    sep_index = input_ids.index(tokenizer_Qna.sep_token_id)

    
    num_seg_a = sep_index + 1

    
    num_seg_b = len(input_ids) - num_seg_a

    
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    
    assert len(segment_ids) == len(input_ids)

    
    
    outputs = Qnamodel(torch.tensor([input_ids]), 
                    token_type_ids=torch.tensor([segment_ids]), 
                    return_dict=True) 

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    
    
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    
    tokens = tokenizer_Qna.convert_ids_to_tokens(input_ids)

    
    answer = tokens[answer_start]

    
    for i in range(answer_start + 1, answer_end + 1):
        
        
        if tokens[i][0:2] == '
            answer += tokens[i][2:]
        
        
        else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')
    return answer

import textwrap


wrapper = textwrap.TextWrapper(width=80) 

intents = json.loads(open('bert_ansrs.json').read())

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
train = pd.read_csv("C:/PythonProj/ChatbotB/train.csv")
classes = train.intent.unique().tolist()

print(classes)
def predict_class2(msg,model):
    print((msg))
    sentence =[msg]
    pred_tokens = map(tokenizer.tokenize, sentence)
    pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

    pred_token_ids = map(lambda tids: tids +[0]*(12-len(tids)),pred_token_ids)
    pred_token_ids = np.array(list(pred_token_ids))
    print("pred_token_ids",pred_token_ids)
    print(pred_token_ids.size)
    predictions = model.predict(pred_token_ids).argmax(axis=-1)
    print("predictions",predictions)
    return predictions

def getResponse2(ints, intents_json, msg):
    for text, label in zip(msg, ints):
        print("text:", text, "\nintent:", classes[label])
        tag= classes[label]
    intents_part = intents["intents"]
    for i in intents_part:
        print(i["tag"], tag)
        if i['tag']==tag:
            
            ansr = answer_question("what is "+msg, i["responses"])
            return ansr
            
    return i["Sorry...Can u repeat"]

def chatbot_response(msg):    
    ints = predict_class2(msg, model)
    res = getResponse2(ints, intents,msg)
    return res


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame
 

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'












@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response





app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


db = SQL("sqlite:///C:/PythonProj/ChatbotB/users.db")

@app.route("/")
@login_required
def home():
    return redirect("/home")

@app.route("/home")
@login_required
def index():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    print("AM in GET")
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    
    session.clear()

    
    if request.method == "POST":

        
        input_username = request.form.get("username")
        input_password = request.form.get("password")

        
        if not input_username:
            return render_template("login.html",messager = 1)

        
        elif not input_password:
             return render_template("login.html",messager = 2)

        
        
        username = db.execute("SELECT * FROM users WHERE username = %s;", (input_username,))

        
        if len(username) != 1 or not check_password_hash(username[0]["hash"], input_password):
            return render_template("login.html",messager = 3)

        
        session["user_id"] = username[0]["id"]


        
        return redirect("/")

    
    else:
        return render_template("login.html")

@app.route("/logout")
def logout():
    """Log user out"""

    
    session.clear()

    
    return redirect("/")



@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    
    if request.method == "POST":

        
        input_username = request.form.get("username")
        input_password = request.form.get("password")
        
        if (len(input_password)<8):
            return render_template("register.html",messager = 6)
        elif not re.search("[a-z]", input_password):
            return render_template("register.html",messager = 7)
        elif not re.search("[A-Z]", input_password):
            return render_template("register.html",messager = 8)
        elif not re.search("[0-9]", input_password):
            return render_template("register.html",messager = 9)
        elif not re.search("[_@$]", input_password):
            return render_template("register.html",messager = 10)
        input_confirmation = request.form.get("confirmation")

        
        if not input_username:
            return render_template("register.html",messager = 1)

        elif not input_password:
            return render_template("register.html",messager = 2)

        
        elif not input_confirmation:
            return render_template("register.html",messager = 4)

        elif not input_password == input_confirmation:
            return render_template("register.html",messager = 3)

        
        username = db.execute("SELECT username FROM users WHERE username = %s;",(input_username,))

        if len(username) == 1:
            return render_template("register.html",messager = 5)

        
        else:
            new_user = db.execute("INSERT INTO users (username, hash) VALUES (:username, :password)",
                                  username=input_username,
                                  password=generate_password_hash(input_password, method="pbkdf2:sha256", salt_length=8),)

            if new_user:
                
                session["user_id"] = new_user

            
            flash(f"Registered as {input_username}")

            
            return redirect("/")

    
    else:
        return render_template("register.html")

@app.route("/facereg", methods=["GET", "POST"])
def facereg():
    session.clear()
    if request.method == "POST":
        encoded_image = (request.form.get("pic")+"==").encode('utf-8')
        username = request.form.get("name")
        name = db.execute("SELECT * FROM users WHERE username = %s;", (username,))
              
        if len(name) != 1:
            return render_template("camera.html",message = 1)

        id_ = name[0]['id']    
        compressed_data = zlib.compress(encoded_image, 9) 
        
        uncompressed_data = zlib.decompress(compressed_data)
        
        decoded_data = b64decode(uncompressed_data)
        
        new_image_handle = open('./static/face/'+str(id_)+'.jpg','wb')
        
        new_image_handle.write(decoded_data)
        new_image_handle.close()
        try:
            image_of_bill = face_recognition.load_image_file('./static/face/'+str(id_)+'.jpg')
        except:
            return render_template("camera.html",message = 5)

        bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]

        unknown_image = face_recognition.load_image_file('./static/face/'+str(id_)+'.jpg')
        try:
            unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        except:
            return render_template("camera.html",message = 2)


        results = face_recognition.compare_faces([bill_face_encoding], unknown_face_encoding)

        if results[0]:
            
            username = db.execute("SELECT * FROM users WHERE username = %s;", ("swa,"))
            session["user_id"] = username[0]["id"]
            return redirect("/")
        else:
            return render_template("camera.html",message=3)


    else:
        return render_template("camera.html")



@app.route("/facesetup", methods=["GET", "POST"])
def facesetup():
    if request.method == "POST":


        encoded_image = (request.form.get("pic")+"==").encode('utf-8')

        

        id_=db.execute("SELECT id FROM users WHERE id = %s,", (session["user_id"])[0]["id"],)
        
        compressed_data = zlib.compress(encoded_image, 9) 
        
        uncompressed_data = zlib.decompress(compressed_data)
        decoded_data = b64decode(uncompressed_data)
        
        new_image_handle = open('./static/face/'+str(id_)+'.jpg', 'wb')
        
        new_image_handle.write(decoded_data)
        new_image_handle.close()
        image_of_bill = face_recognition.load_image_file('./static/face/'+str(id_)+'.jpg')    
        try:
            bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]
        except:    
            return render_template("face.html",message = 1)
        return redirect("/home")

    else:
        return render_template("face.html")

def gen_frames():  
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(thres):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA) 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mask, frame = cv2.threshold(gray, 120,255,cv2.THRESH_BINARY)
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
            
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass

@app.route('/filters')
def filters():
    return render_template('filters.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('thres') == 'Threshold':
            global thres
            thres=not thres
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('filters.html')
    return render_template('filters.html')



def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return render_template("error.html",e = e)



for code in default_exceptions:
    app.errorhandler(code)(errorhandler)


if __name__ == "__main__":
    app.run()
