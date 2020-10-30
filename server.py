from flask import Flask, render_template, request, send_file, Response
from PIL import Image
from io import BytesIO
from numpy import fromfile, uint8

from image_resizer import resize
from mnist_mlp import load_keras_model, predict_number

model = load_keras_model()
app = Flask(__name__, template_folder="./templates/", static_url_path="/images", static_folder="images")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/healthz", methods=["GET"])
def healthCheck():
    return "", 200

@app.route("/image", methods = ['GET','POST'])
def image_resize():
    if request.method == "POST":
        width, height = 28, 28
        try:
            source = Image.open(request.files['source'])
            adjusted_image = resize(source, width, height)
            result = predict_number(model, adjusted_image, width, height)
        except Exception as e:
            print("error : %s" % e)
            return Response("fail", status=400)

    return str(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='80', debug=True)