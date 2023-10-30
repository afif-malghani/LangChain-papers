from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from prompts import qa_template
from huggingface_hub.utils._errors import RepositoryNotFoundError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.llms import CTransformers
from huggingface_hub import hf_hub_download

from config import (
    papersDirectory,
    papersFilesType,
    splitterChunkSize,
    splitterChunkOverlap,
    encoderModel,
    encoderKwargs,
    vectorstoreCollectionName,
)


# Wrap prompt template in a PromptTemplate object
# def set_qa_prompt():
#     prompt = PromptTemplate(
#         template=qa_template, input_variables=["context", "query", "history"]
#     )
#     return prompt


def db_build():
    # Load PDF file from data path
    loader = DirectoryLoader(
        papersDirectory, glob=papersFilesType, loader_cls=PyPDFLoader
    )
    documents = loader.load()

    # Split text from PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=splitterChunkSize, chunk_overlap=splitterChunkOverlap
    )
    texts = text_splitter.split_documents(documents)

    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name=encoderModel,
        model_kwargs=encoderKwargs,
    )

    # Build and persist FAISS vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(vectorstoreCollectionName)
    
    return vectorstore



def build_conversational_chain(llm, vectordb, memory):
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    print(memory)
    dbqa = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(), memory=memory)
    return dbqa


# Build RetrievalQA object
def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return dbqa



def load_model():
    
    # check if file exists
    try:    
        llm = CTransformers(
            model="llama-2-7b-chat.ggmlv3.q8_0.bin",  # Location of downloaded GGML model
            model_type="llama",  # Model type Llama
            config={"max_new_tokens": 256, "temperature": 0.01},
        )
        
    # if huggingface_hub.utils._errors.RepositoryNotFoundError
    except RepositoryNotFoundError as e:
        print(
            "\nFailed to load model: \n",
            e
        )
            
        
        
    except Exception as e:
        print("\nFailed to load model: \n", e, "\nPlease download the model in the root directory of the project\n")
        exit()
        
    return llm


# Instantiate QA object
def setup_dbqa(mem_key):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    try:
        vectordb = FAISS.load_local("vectorstore/db_faiss", embeddings)
    except:
        vectordb = db_build()
    # qa_prompt = set_qa_prompt()
    llm = load_model()
    # dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)
    memory = ConversationBufferMemory(memory_key=mem_key, return_messages=True)
    # print(memory)
    dbqa = build_conversational_chain(llm, vectordb, memory=memory)

    return dbqa
