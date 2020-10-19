import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask,request,render_template 
from werkzeug.utils import secure_filename
import cv2
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model(r"C:\Users\adars\Downloads\Best.hdf5")
video = cv2.VideoCapture(0)
label_dict={0:'Mask',1:'No Mask'}

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        while True:
            _, frame = video.read()
            im = Image.fromarray(frame, 'RGB')
            im = im.resize((224,224))
            x = img_to_array(im)
            images = np.array([x],dtype="float32")
            classes = model.predict(images)
            classes = labels_dict[np.argmax(classes)]
            if classes == 'No Mask':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
            cv2.imshow("Capturing", frame)
            key=cv2.waitKey(1)
            if key == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()
        return None
    return None
    

if __name__ == "__main__":
    app.run(debug=True)
