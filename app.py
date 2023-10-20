from flask import request, render_template

from flask_cors import CORS, cross_origin

from flask import Flask

from flask_restful import Resource, Api

from routes.index import index


from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplittter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.document_loaders import PyPDFDirectoryLoader

import json


app = Flask(__name__)
api = Api(app)

api.add_resource(index, '/')

if __name__ == '__main__':
    filePath = "config.json"
    with open(filePath, "r") as json_file:
        constants = json.load(json_file)
    if constants:
        loader = PyPDFDirectoryLoader(constants["papersDirectory"])

        docs = loader.load()

        r_splitter = RecursiveCharacterTextSplittter(
                chunk_size = constants["R_SPLITTER_CHUNK_SIZE"],
                chunk_overlap = constants["R_SPLITTER_CHUNK_OVERLAP"],
                seperators = constants["R_SPLITTER_SEPERATOR"]
                )

        split_docs = r_splitter.split_documents(docs)

        hf_embed = HuggingFaceEmbeddings(
                model_name= constants["ENCODER_MODEL"],
                encode_kwargs = constants["ENCODER_KWARGS"]
                )
        db = Chroma(constants["CHROMA_COLLECTION_NAME"], hf_embed)
        db.add_documents(split_docs)
        db.persist()

        llm = HuggingFacePipeline.from_model_id(
                model_id=constants["LLM_MODEL"],
                taks = constants["LLM_TASK"],
                pipeline_kwargs = constants["pipeline_kwargs"]
                )
        qa_chain = RetrievalQA.from_chain_type(llm, retriever = db.as_retriever())

        app.run(host='0.0.0.0', debug=True, port=7777)
