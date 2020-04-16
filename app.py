
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import keras
import tensorflow as tf
import json
import flask

from flask import Flask,render_template,request
from keras.models import model_from_json
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from keras import backend as K
from random import gauss
from random import seed


sess = K.get_session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)


#loading the model
with open('model/OddOrRotten.json','r') as f:
    model_json = json.load(f)
    
jtopy=json.dumps(model_json)
model = model_from_json(jtopy)
model.load_weights('model/OddOrRotten.h5')


app = Flask(__name__)


def model_predict(img_path):
    
    # Preprocessing the image
    img_pre=image.load_img(img_path,target_size=(150,150))
    img_pre=image.img_to_array(img_pre)
    img_pre=np.expand_dims(img_pre,axis=0)
    
    # predicting model
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        result=model.predict(img_pre)

    if result[0][0]<=0:
        string = 'odd'
    else :
        string = 'rotten' 
    
    classified_prob = result[0][0] if result[0][0] >= 0.5 else 1 - result[0][0]
    
    return string , classified_prob


@app.route('/')
def student():
    return render_template('index.html')




@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        
        # Save the file to ./uploads
        # Making the image path unique
        seed(f)
        rand = gauss(0, 100)
        file_path = os.path.join( "static/uploads", str(rand)+secure_filename(f.filename))
        f.save(file_path)
        
        #calling model_predict function
        answer , prob = model_predict(file_path)
        
        prob = round((prob * 100), 2)
        
        return render_template('result.html',result = answer,file_path = file_path,prob = prob)
    return None

if __name__ == '__main__':
    app.run(debug = True)
    