from flask import make_response
from flask_restful import Resource

class index(Resource):
    def get(self):
        return(make_response("API is up and running"))
