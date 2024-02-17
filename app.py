from flask import Flask, render_template, request, jsonify

import os

from test1 import chatbot
from dotenv import load_dotenv

load_dotenv()
bot =  chatbot()


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    result = get_Chat_response(input)
    print('input: ', input) 
    print('result: ', result) 

    return result


def get_Chat_response(text):
    return bot.chat_public(text)

   

if __name__ == '__main__':
    app.run()
