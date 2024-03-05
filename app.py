from flask import Flask, render_template, request, jsonify

from test1 import chatbot
from dotenv import load_dotenv
import awsgi

load_dotenv()
bot = chatbot()


app = Flask(__name__)
app.config.from_pyfile("settings.py")

#AWS
def handler(event, context):
    return awsgi.response(app, event, context)



#APP

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    query = request.form["msg"]
    _response = get_Chat_response(query)
    print(f"User query: {query}")
    print(f"Response: {_response}")
    return _response


def get_Chat_response(text):
    return bot.chat_public(text)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
