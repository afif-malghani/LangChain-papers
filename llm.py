# import
from langchain.llms import HuggingFacePipeline
from huggingface_hub import hf_hub_download

# Local CTransformers wrapper for Llama-2-7B-Chat
from langchain.llms import CTransformers

# Local CTransformers wrapper for Llama-2-7B-Chat

try:
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",  # Location of downloaded GGML model
        model_type="llama",  # Model type Llama
        config={"max_new_tokens": 256, "temperature": 0.01},
    )
except FileNotFoundError as e:
    print(
        "\nFailed to load model: \n",
        e,
        "\nDownlaoding llama-2-7b-chat.ggmlv3.q8_0.bin \n",
    )
    hf_hub_download(
        repo_id="TheBloke/Llama-2-7B-Chat-GGML",
        filename="llama-2-7b-chat.ggmlv3.q8_0.bin",
        local_dir="models",
    )

except Exception as e:
    print("Failed to load model: \n", e)