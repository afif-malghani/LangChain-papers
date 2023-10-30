from flask import request, render_template, make_response

from flask_cors import CORS, cross_origin

from flask import Flask

from flask_restful import Resource, Api

from utils import setup_dbqa

import os
from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename

import json



UPLOAD_FOLDER = 'static/pdf'
ALLOWED_EXTENSIONS = {'pdf'}


dbqa = None

chat_history = []


class Query(Resource):
    def post(self):
        data = request.get_json()
        response = dbqa({"question": data["query"], "chat_history": chat_history})
        
        chat_history.append((data["query"], response["answer"]))
        return response["answer"]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class upload_file(Resource):
    def post(self):
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            dbqa = setup_dbqa()
        return redirect(url_for('chatbot', pdf_filename=filename))
        # return make_response(render_template("chatbot.html"))

class index(Resource):
    def get(self):
        return make_response(render_template("index.html"))


class chatbot(Resource):
    def get(self):
        pdf_filename = request.args.get('pdf_filename')
        if(pdf_filename == None):
            return flash('No pdf file uploaded')
        return make_response(render_template("chatbot.html", pdf_filename=pdf_filename))

app = Flask(__name__)
api = Api(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


api.add_resource(index, "/")
api.add_resource(Query, "/query")
api.add_resource(upload_file, "/upload")
api.add_resource(chatbot, "/chat")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=7777)
