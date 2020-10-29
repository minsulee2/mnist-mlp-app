from flask import Flask, render_template, request, send_file, Response
# from PIL import Image
from cv2 import imdecode, IMREAD_UNCHANGED
from io import BytesIO
from PIL import Image
from numpy import fromfile, uint8

from image_resizer import resize

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
        try:
            # source = Image.open(request.files['source'].stream).convert('RGB')
            img = imdecode(fromfile(request.files['source'], uint8), IMREAD_UNCHANGED)
        except Exception as e: 
            print("error : %s" % e)
            return Response("fail", status=400)

        resultImage = resize(img, 28, 28)
        # cv2 image convert to PIL image 
        # and PIL image to bytes 
        im_pil = Image.fromarray(img)
        img_io = BytesIO()
        im_pil.save(img_io, 'PNG')
        img_io.seek(0)

    return '2'
    # return send_file(img_io, mimetype="image/jpeg")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='80', debug=True)