from flask import Flask, render_template, request, jsonify

import os

from chatbot import chatbot
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
    print('input: ', input) #xin chào
    print('result: ', result) #Xin chào! Tôi là trợ lý ảo của BKTechStore. Tôi có thể giúp bạn với những câu hỏi về trang web BKTechStore. Bạn có câu hỏi gì cần giải đáp không ạ?

    return result


def get_Chat_response(text):
    return bot.chat_public(text)

   

if __name__ == '__main__':
    app.run()
