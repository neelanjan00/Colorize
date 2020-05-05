from flask import Flask, request, redirect, render_template
import numpy as np
import cv2, base64
import io
from PIL import Image
import subprocess
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

op = subprocess.run(['bash','down.sh'],check=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
print(op)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

prototxt = "./model/colorization_deploy_v2.prototxt"
model = "./model/colorization_release_v2.caffemodel"
points = "./model/pts_in_hull.npy"

net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

@app.route('/', methods=['GET', 'POST'])
def base():
    if request.method == 'GET':
        return render_template("base.html", check = 0)
    else:
        if 'InputImg' not in request.files:
            print("No file part")
            return redirect(request.url)
        file = request.files['InputImg']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filestr = request.files['InputImg'].read()
            npimg = np.fromstring(filestr, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            scaled = image.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50

            net.setInput(cv2.dnn.blobFromImage(L))
            ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

            L = cv2.split(lab)[0]
            colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
            colorized = np.clip(colorized, 0, 1)
            colorized = (255 * colorized).astype("uint8")
            
            pil_img = Image.fromarray(colorized)
            buff = io.BytesIO()
            pil_img.save(buff, format="JPEG")
            new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")

        return render_template('base.html', img = new_image_string, check = 1)

if __name__ == '__main__':
    app.secret_key = 'qwertyuiop1234567890'
    port = int(os.environ.get('PORT', 33507))
    print(port)
    app.run(debug=True,host='0.0.0.0',port=port)
    print("APP IS RUNNING")
