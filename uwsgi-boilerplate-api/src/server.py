from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/ping")
def ping():
    return "pong"

@app.route("/")
def homepage():
	return "Welcome to the hello world REST API!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
