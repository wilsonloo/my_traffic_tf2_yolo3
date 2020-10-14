import tensorflow as tf
import numpy
import os
import io
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from tensorflow import keras
from flask import Flask, request
import image_demo

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True
with tf.compat.v1.Session(config=config) as sess:
    print("gpu set")    

app = Flask(__name__)

# 进行分类
# @param iamgeData 尺寸为192*192 的图片
def classify_image(model, imageData, toDecode):
    fixedImage = preprocess_image(imageData, toDecode)
    inputBatch = fixedImage[np.newaxis, ...]
    predict = model.predict(inputBatch)
    possiables = predict[0]
    category, confidence = get_category_from_possiable(possiables)
    return category, confidence

@app.route('/')
def index():
    return "hello world"

@app.route('/classify_traffic', methods=['POST'])
def req_classify_traffic():
    if request.method == 'POST':
        binaryFlag = request.args.get('binary')
        toDecode = False
        imageBinary = request.get_data() 
        image = cv2.imdecode(np.frombuffer(imageBinary, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        bboxes, original_image = image_demo.predicate(image, None)
        bboxesList = []
        for box in bboxes:
            print(111, box)
            bboxesList.append(":".join(map(str, box))) 
        print(bboxesList)
        return "|".join(bboxesList)

if __name__ == "__main__":
    app.run(host="192.168.1.19", port=5000, debug=True)
