# import
from langchain.llms import HuggingFacePipeline


# Local CTransformers wrapper for Llama-2-7B-Chat
from langchain.llms import CTransformers

# Local CTransformers wrapper for Llama-2-7B-Chat

try:
    llm = CTransformers(
        model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",  # Location of downloaded GGML model
        model_type="llama",  # Model type Llama
        config={"max_new_tokens": 256, "temperature": 0.01},
    )
except Exception as e:
    print(
        "\nFailed to load model: \n",
        e,
        "\nMake sure that a model.bin file is peresent in the models directory.\n",
    )
