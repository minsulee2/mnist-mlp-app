from flask import Flask, render_template, request, Response
app = Flask(__name__, template_folder="./templates/")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/healthz", methods=["GET"])
def healthCheck():
    return "", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='80', debug=True)